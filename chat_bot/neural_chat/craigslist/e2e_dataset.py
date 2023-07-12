import json
import torch
import sys

sys.path.append("./")
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from transformers.trainer_pt_utils import LabelSmoother
from chat_bot.neural_chat.conversation import get_default_conv_template
from typing import Dict

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class SimpleDialogDataset(Dataset):
    """
    강화학습 없이 end-to-end 챗봇을 훈련하기 위한 데이터셋입니다.
    상품 가격을 별도의 토큰으로 변환하지 않고 그대로 학습합니다.
    """

    def __init__(
        self, fp: str, tokenizer: PreTrainedTokenizerFast, block_size: int = 256
    ):
        with open(fp, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        conv = get_default_conv_template()
        roles = {"구매자": conv.roles[0], "판매자": conv.roles[1]}
        data = []
        for d in raw_data:
            conv.messages = []
            conv.scenario["제목"] = d["title"]
            conv.scenario["상품 설명"] = d["description"]
            conv.scenario["가격"] = d["price"]
            for ev in d["events"]:
                conv.append_message(role=roles[ev["role"]], message=ev["message"])
            data.append(conv.get_prompt())

        data = tokenizer.eos_token.join(data)
        self.tokens = tokenizer.encode(data)
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) // self.block_size - 1

    def __getitem__(self, i):
        tokens = self.tokens[i * self.block_size :][: self.block_size]
        data = self.tokenizer.prepare_for_model(
            tokens, return_tensors="pt", return_token_type_ids=False
        )
        data["labels"] = data["input_ids"].clone()
        return data


class VicunaDialogDataset(Dataset):
    """Vicuna의 학습 방법을 따라서 챗봇의 발화를 제외한 텍스트는 masking하는 데이터셋입니다."""

    def __init__(self, fp: str, tokenizer: PreTrainedTokenizerFast):
        with open(fp, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        conv = get_default_conv_template()
        roles = {"구매자": conv.roles[0], "판매자": conv.roles[1]}
        conversations = []
        for i, d in enumerate(raw_data):
            conv.messages = []
            conv.scenario["제목"] = d["title"]
            conv.scenario["상품 설명"] = d["description"]
            conv.scenario["가격"] = d["price"]

            if roles[d["events"][0]["role"]] != conv.roles[0]:
                d["events"] = d["events"][1:]

            for j, ev in enumerate(d["events"]):
                role = roles[ev["role"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role=roles[ev["role"]], message=ev["message"])
            conversations.append(conv.get_prompt())

        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
        ).input_ids
        targets = input_ids.clone()

        # target을 masking해서 챗봇의 발화에서만 loss를 계산합니다.
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep2)
            cur_len = 0
            for turn in turns:
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                instruction_len = len(tokenizer(parts[0]).input_ids)

                # Ignore the user instructions
                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                cur_len += turn_len

            target[cur_len:] = IGNORE_TOKEN_ID

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        self.data = {
            "input_ids": input_ids,
            "labels": targets,
            "attention_mask": input_ids.ne(tokenizer.pad_token_id),
        }

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.data["input_ids"][index],
            labels=self.data["labels"][index],
            attention_mask=self.data["attention_mask"][index],
        )


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-5.8b")
    tokenizer.model_max_length = 1024
    ds = VicunaDialogDataset("./data/chatbot_train.json", tokenizer)
    print(ds[0])
