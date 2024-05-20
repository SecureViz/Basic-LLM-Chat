# A basic command-line chat with an LLM

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import os

# ------------------------ Configuration ------------------------

MODEL_PATH = "../Meta-Llama-3-8B-Instruct" # The path for the folder that the LLM is in
MODEL_NAME = "Llama 3" # The name that is used when the LLM finishes generating a chat
DEBUG = False   # Shows debug information

systemPrompt = ( # The prompt that tells the AI how to behave when given a prompt. You can split this into multiple lines by using a set of quotes on each line 
                 # Use \n at the end of each line for the LLM to know they are new lines)
    "You are a helpful assistant named Llama. Use a friendly, casual tone in your responses."
    ) 

# ------------------------ Main Script ------------------------

clear = lambda: os.system('cls')

# Prepare the input as before
chat = [
    {"role": "system", "content": systemPrompt}
]

# 1: Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", load_in_4bit=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
streamer = TextStreamer(tokenizer)

def generate_output():
    clear()
    for message in chat:
        if (message["role"] == "user"):
            print("User:")
        elif (message["role"] == "assistant"):
            print(f"{MODEL_NAME}:")
        else:
            continue

        print(message["content"] + "\n")

while True:
    message = input("User: ")
    chat.append(
        { "role": "user", "content": message }
    )

    # 2: Apply the chat template
    formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    if (DEBUG):
        print("Formatted chat:\n", formatted_chat)

    # 3: Tokenize the chat (This can be combined with the previous step using tokenize=True)
    inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
    # Move the tokenized inputs to the same device the model is on (GPU/CPU)
    inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
    if (DEBUG):
        print("Tokenized inputs:\n", inputs)

    # 4: Generate text from the model
    print(f"\n{MODEL_NAME}:\n")
    print("model")
    outputs = model.generate(**inputs, streamer=streamer, max_new_tokens=512, temperature=1.0)
    if (DEBUG):
        print("Generated tokens:\n", outputs)

    # 5: Decode the output back to a string
    decoded_output = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
    chat.append(
        { "role": "assistant", "content": decoded_output }
    )

    if (DEBUG):
        print("Decoded output:\n", decoded_output)
    else:
        generate_output()
