import os
import sys
from pathlib import Path
import json

path = Path(__file__).parent.parent
path = os.path.join(path, "chat_bot")
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
def convert_to_model_input(data: dict):
    conv = get_default_conv_template()

    conv.scenario["제목"] = data['title']
    conv.scenario["상품 설명"] = data['description']
    conv.scenario["가격"] = data['price']
    conv.messages = data['events']
    conv.append_message(conv.roles[1], "")
    # print(f"model_input:{model_input}")
    return conv
