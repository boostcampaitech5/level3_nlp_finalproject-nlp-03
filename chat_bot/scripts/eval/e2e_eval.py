import sys

sys.path.append("./chat_bot")
from neural_chat.e2emodel.e2e_lora_model import E2ELoRA
from neural_chat.e2emodel.preprocess import format_scenario
import argparse
import torch
import random
import json


def rollout(model: E2ELoRA, scenario: str):
    scenario["events"] = []
    info = format_scenario(scenario, model.tokenizer.sep_token)
    print(info.replace(model.tokenizer.sep_token, "\n"))
    while True:
        user_input = input()
        if user_input == "quit":
            break
        scenario["events"].append({"role": "구매자", "message": user_input})
        model_response = model.generate(scenario)
        print(model_response)
        scenario["events"].append({"role": "판매자", "message": model_response})


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
    args.model_checkpoint_dir = "/opt/ml/level3_nlp_finalproject-nlp-03/chat_bot/logs/polyglot-12.8b/checkpoint-300"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = E2ELoRA(args.model_checkpoint_dir, device=device, do_quantize=True)

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i in range(args.num_rollouts):
        print(f"rollout #{i + 1}")
        scenario = random.choice(data)
        rollout(model, scenario)
