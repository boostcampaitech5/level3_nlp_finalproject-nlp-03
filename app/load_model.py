import os 
import sys
from pathlib import Path
import json
path = Path(__file__).parent.parent
path = os.path.join(path, "chat_bot/neural_chat/e2emodel/")
print(path)
sys.path.append(path)
from e2e_lora_model import E2ELoRA
import re 


def load_model():
    lora = E2ELoRA(
    "/opt/ml/level3/chat_bot/logs/checkpoint-300",
    "cuda",
    )
    return lora

def convert_to_model_input(chat):
    messages = chat.content.strip().split("\n")
    events = []
    for message in messages:
        dataset = {"role" : message[:3], "message" : message[4:]}
        events.append(dataset)
    dialogue = {
                "title" : chat.product.title,
                "description" : chat.product.description,
                "price" : float(chat.product.price),
                "events" :  events
                }
    print(f"model_input:{dialogue}")
    return dialogue


