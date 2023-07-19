import os
import sys
from pathlib import Path

path = Path(__file__).parent.parent
path = os.path.join(path, "chat_bot")
sys.path.append(path)
from chat_bot.neural_chat.e2emodel.e2e_lora_model import E2ELoRA
from chat_bot.neural_chat.conversation import get_default_conv_template


def load_model():
    lora = E2ELoRA(
        "ggul-tiger/kullm-12.8b-negobot-372data",
        "cuda",
    )
    return lora


# model.generate( ) 에 들어가는 입력 포맷
def convert_to_model_input(data: dict):
    conv = get_default_conv_template()
    conv.load_dict(data)
    conv.append_message(conv.roles[1], "")
    return conv
