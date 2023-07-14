import os 
import sys
from pathlib import Path
import json
path = Path(__file__).parent.parent
path = os.path.join(path, "chat_bot")
print(path)
sys.path.append(path)
from chat_bot.neural_chat.e2e_lora_model import E2ELoRA
import re 


def load_model():
    lora = E2ELoRA(
    "/opt/ml/level3/chat_bot/logs/checkpoint-87",
    "cuda",
    )
    return lora

# model.generate( ) 에 들어가는 입력 포맷
def convert_to_model_input(chat):
    messages = chat.content.strip().split("\n")
    events = []
    for message in messages:
        dataset = {"role" : message[:3], "message" : message[4:]}
        events.append(dataset)
    model_input = {
                "title" : chat.product.title,
                "description" : chat.product.description,
                "price" : float(chat.product.price),
                "events" :  events
                }
    # print(f"model_input:{model_input}")
    return model_input


