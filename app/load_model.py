import os
import sys
from pathlib import Path
import json

path = Path(__file__).parent.parent
path = os.path.join(path, "chat_bot")
print(path)
sys.path.append(path)
from chat_bot.neural_chat.e2emodel.e2e_lora_model import E2ELoRA
from chat_bot.neural_chat.conversation import get_default_conv_template
import re


def load_model():
    lora = E2ELoRA(
        "/opt/ml/level3/chat_bot/logs/checkpoint-87",
        "cuda",
    )
    return lora


# model.generate( ) 에 들어가는 입력 포맷
def convert_to_model_input(chat):
    conv = get_default_conv_template()
    messages = chat.content.strip().split("\n")
    events = []
    for message in messages:
        dataset = [message[:3], message[4:]]
        events.append(dataset)

    conv.scenario["제목"] = chat.product.title
    conv.scenario["상품 설명"] = chat.product.description
    conv.scenario["가격"] = chat.product.price
    conv.messages = events
    conv.append_message(conv.roles[1], "")
    # print(f"model_input:{model_input}")
    return conv
