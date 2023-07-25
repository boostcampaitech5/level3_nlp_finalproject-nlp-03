import os
import sys

path = "/opt/level3_nlp_finalproject-nlp-03"
sys.path.append(path)
print(path)

from chat_bot.neural_chat.e2emodel.e2e_lora_model import E2ELoRA
from chat_bot.neural_chat.conversation import get_conv_template


def load_model():
    lora = E2ELoRA(
        "ggul-tiger/negobot_361_v3",
        "cuda",
    )
    return lora

# model.generate( ) 에 들어가는 입력 포맷
def convert_to_model_input(data: dict):
    conv = get_conv_template("v2")
    conv.load_dict(data)
    conv.append_message(conv.roles[1], "")
    return conv
