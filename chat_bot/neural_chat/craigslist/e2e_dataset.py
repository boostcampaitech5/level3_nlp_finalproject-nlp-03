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
        data = []
        for d in raw_data:
            if d["events"][0]["role"] != conv.roles[0]:
                d["events"] = d["events"][1:]

            conv.load_dict(d)
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
        self.data = []

        with open(fp, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        conv = get_default_conv_template()
        conversations = []
        for d in raw_data:
            if d["events"][0]["role"] != conv.roles[0]:
                d["events"] = d["events"][1:]

            conv.load_dict(d)
            conversations.append(conv.get_prompt())

        # target을 masking해서 챗봇의 발화에서만 loss를 계산합니다.
        sep = conv.sep + conv.roles[1] + ": "
        for conversation in conversations:
            input_ids = []
            target = []
            turns = conversation.split(conv.sep2)
            for turn in turns:
                if turn == "":
                    break

                parts = turn.split(sep)
                if len(parts) != 2:
                    break

                parts[0] += sep
                parts[1] += conv.sep2

                parts[0] = tokenizer.encode(parts[0])
                parts[1] = tokenizer.encode(parts[1])

                input_ids += parts[0]
                input_ids += parts[1]
                target += [IGNORE_TOKEN_ID] * len(parts[0])
                target += parts[1]

            target += [IGNORE_TOKEN_ID] * (tokenizer.model_max_length - len(target))
            tokens = tokenizer.prepare_for_model(
                input_ids,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
            )
            tokens["labels"] = torch.tensor(target)
            self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("nlpai-lab/kullm-polyglot-12.8b-v2")
    tokenizer.model_max_length = 1024
    ds = VicunaDialogDataset("./data/price_parsed_train.json", tokenizer)
    print(ds[0])
