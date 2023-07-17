from typing import Tuple
from transformers import (
    GPT2LMHeadModel,
    AutoModelForCausalLM,
    GPT2TokenizerFast,
    # PretrainedTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from torch.utils.data import Dataset
import argparse
from neural_chat.gpt2 import format_event
from neural_chat.craigslist import Craigslist
import os


######################
# CRAIGSLIST DATASET #
######################


class DialogDataset(Dataset):
    def __init__(
        self, cg: Craigslist, tokenizer: GPT2TokenizerFast, block_size: int = 256
    ):
        # get dialog
        data = []
        for scene in cg:
            data.append(format_event(scene.events[-1]))

        data = tokenizer.eos_token.join(data)
        self.tokens = tokenizer.encode(data)
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) // self.block_size - 1

    def __getitem__(self, i):
        tokens = self.tokens[i * self.block_size :][: self.block_size]
        data = self.tokenizer.prepare_for_model(tokens, return_tensors="pt")
        data["labels"] = data["input_ids"].clone()
        return data


def make_craigslist_dataset(
    tokenizer: GPT2TokenizerFast,
    train_fp: str,
    val_fp: str,
) -> Tuple[DialogDataset, DialogDataset]:
    # make data
    cg_train = Craigslist(train_fp)
    cg_val = Craigslist(val_fp)

    # train and test
    dd = lambda d: DialogDataset(d, tokenizer=tokenizer)
    return dd(cg_train), dd(cg_val)


###############
# MAIN SCRIPT #
###############

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt2-type", default="heegyu/ajoublue-gpt2-medium-dialog")
    parser.add_argument("--train-fp", default="/opt/ml/chai-naacl-2022/data/fixed_final_translated_train.json")
    parser.add_argument("--val-fp", default="/opt/ml/chai-naacl-2022/data/fixed_final_translated_dev.json")
    parser.add_argument("--output-dir", default="./gpt2_medium")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0002)
    args = parser.parse_args()

    # make tokenizer
    token = GPT2TokenizerFast.from_pretrained(args.gpt2_type)
    token.add_tokens(["$PRICE", "$PARTNER_PRICE", "<sep>"])
    # token.add_special_token({})

    # make dataset
    train_dataset, val_dataset = make_craigslist_dataset(
        tokenizer=token, train_fp=args.train_fp, val_fp=args.val_fp
    )

    # initialize model
    model = GPT2LMHeadModel.from_pretrained(args.gpt2_type)
    model.resize_token_embeddings(len(token))

    # finetune
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=args.fp16,
        no_cuda=False,
        num_train_epochs=20,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        warmup_steps=100,
        dataloader_drop_last=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="wandb",
    )
    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=token,
        data_collator=default_data_collator,
    )

    trainer.train()

    model.save_pretrained(os.path.join(args.output_dir,"best_model"))
    token.save_pretrained(os.path.join(args.output_dir,"best_model"))
