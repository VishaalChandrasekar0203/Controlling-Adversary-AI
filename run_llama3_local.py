import requests
import json

url = "http://localhost:11434/api/chat"

def llama3(prompt):
    data = {
        "model": "llama3",
        "messages": [
            {
              "role": "user",
              "content": prompt
            }
        ],
        "stream": False
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    return(response.json()['message']['content'])


response = llama3("what is Quantum feild theory, explain it to a child ")
print(response)


trl sft \
--model_name_or_path meta-llama/Meta-Llama-3-8B \
--dataset_name HuggingFac \
--learning_rate 0.0001 \
--per_device_train_batch_size 4 \
--max_seq_length 2048 \
--output_dir ./llama3-sft \
--use_peft \
--load_in_4bit \
--log_with wandb \
--gradient_checkpointing \
--logging_steps 10