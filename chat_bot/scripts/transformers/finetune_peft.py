from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    default_data_collator,
)
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    TaskType,
)
from torch.utils.data import Dataset
import argparse
import json


# simple dataset for end-to-end negobot
class SimpleDialogDataset(Dataset):
    """
    강화학습 없이 end-to-end 챗봇을 훈련하기 위한 데이터셋입니다.
    상품 가격을 별도의 토큰으로 변환하지 않고 그대로 학습합니다.
    """

    def __init__(self, fp: str, tokenizer: AutoTokenizer, block_size: int = 256):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        data = [self.dialog_formatter(d, tokenizer.sep_token) for d in data]
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

    def dialog_formatter(self, scence: dict, sep_token: str) -> str:
        title = f"제목: {scence['scenario']['kbs'][0]['item']['Title']}"
        desc = "\n".join(scence["scenario"]["kbs"][0]["item"]["Description"])
        desc = f"상품 설명: {desc}"
        price = f"가격: {scence['scenario']['kbs'][0]['item']['Price']}"
        chats = self.chat_formatter(scence["events"], sep_token)

        formatted_dialogue = sep_token.join([title, desc, price, chats])
        return formatted_dialogue

    def chat_formatter(self, events: dict, sep_token: str) -> str:
        chats = []
        for event in events:
            if event["action"] == "message":
                chats.append(event["data"])
            elif event["action"] == "offer":
                chats.append(f"offer {event['data']['price']}")
            elif event["action"] == "accept":
                chats.append("accept")
            elif event["action"] == "reject":
                chats.append("reject")
            elif event["action"] == "quit":
                chats.append("quit")
            else:
                raise NotImplementedError
        agent_mapping = ["구매자", "판매자"]
        chats = [
            f"{agent_mapping[events[i]['agent']]}: {chat}"
            for i, chat in enumerate(chats)
        ]

        return sep_token.join(chats) + sep_token


###############
# MAIN SCRIPT #
###############
def train(args):
    # make tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.sep_token = "<|sep|>"
    tokenizer.pad_token = tokenizer.eos_token

    # make dataset
    train_dataset = SimpleDialogDataset(
        args.train_fp, tokenizer=tokenizer, block_size=256
    )

    # QLoRA configs
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    if args.peft_type=="lora":
        peft_config = LoraConfig(
            r=args.lora_r,
            # target_modules=["query_key_value"],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            bias="none", # https://huggingface.co/docs/peft/task_guides/token-classification-lora#:~:text=For%20performance%2C%20we%20recommend%20setting%20bias%20to%20None%20first%2C%20and%20then%20lora_only%2C%20before%20trying%20all.
        )
    elif args.peft_type=="prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=args.n_virtual_token,
        )

    # initialize model
    print("load model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config
    )

    print("get peft model...")
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    print(model)
    model.print_trainable_parameters()

    # finetune
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epoch,
        max_steps=args.max_steps,
        save_steps=10,
        warmup_steps=5,
        logging_steps=5,
        dataloader_drop_last=True,
        # report_to="wandb",
        # push_to_hub=True,
    )

    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()

    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-fp",
        default="/opt/ml/level3_nlp_finalproject-nlp-03/data/cherrypick_train.json",
    )

    parser.add_argument("--model-name", default="EleutherAI/polyglot-ko-5.8b")
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="./chat_bot/logs/kullm-polyglot-12.8b")

    # peft methods
    parser.add_argument("--peft-type", default="lora") # ["lora", "prefix"]
    # lora
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    # prefix tuning
    parser.add_argument("--n_virtual_token", type=int, default=25)
    args = parser.parse_args()

    train(args)
