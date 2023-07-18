import sys

sys.path.append("./")
from chat_bot.neural_chat.e2emodel.e2e_lora_model import E2ELoRA
from chat_bot.neural_chat.conversation import get_default_conv_template
from transformers import GenerationConfig
import argparse
import torch
import random
import json


def rollout(model: E2ELoRA, scenario: str, gen_config: GenerationConfig):
    conv = get_default_conv_template()
    conv.scenario["제목"] = scenario["title"]
    conv.scenario["상품 설명"] = scenario["description"]
    conv.scenario["가격"] = scenario["price"]
    print(conv.get_scenario())
    while True:
        user_input = input()
        if user_input == "quit":
            break
        conv.append_message("구매자", user_input)
        conv.append_message("판매자", "")
        model_response = model.generate(conv, gen_config)
        print(model_response)
        conv.update_last_message(model_response)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data-path", required=True)
    # parser.add_argument("--output-path", required=True)
    parser.add_argument("--model_checkpoint_dir")
    parser.add_argument("--num-rollouts", type=int, default=30)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    args.data_path = "./data/new_format_dev.json"
    args.model_checkpoint_dir = (
        "/opt/ml/level3_nlp_finalproject-nlp-03/chat_bot/logs/kullm-12.8b/checkpoint-80"
    )

    gen_config = GenerationConfig(
        max_new_tokens=128,
        use_cahce=False,
        early_stopping=True,
        do_sample=True,
        top_k=100,
        top_p=0.85,
        num_beams=5,
        temperature=0.9,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = E2ELoRA(args.model_checkpoint_dir, device=device, do_quantize=True)

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i in range(args.num_rollouts):
        print(f"rollout #{i + 1}")
        scenario = random.choice(data)
        rollout(model, scenario, gen_config)