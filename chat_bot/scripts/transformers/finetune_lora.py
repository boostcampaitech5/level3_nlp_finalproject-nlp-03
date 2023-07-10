from typing import Tuple
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    AutoTokenizer,
    LlamaTokenizerFast,
    LlamaForCausalLM,
    # PretrainedTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    LoraModel,
    get_peft_config,
    get_peft_model,
    TaskType,
    PeftConfig,
    PeftModelForCausalLM,
)
from torch.utils.data import Dataset
import argparse
from neural_chat.gpt2 import format_event
from neural_chat.craigslist import Craigslist
import os
# from huggingface_hub import interpreter_login

######################
# CRAIGSLIST DATASET #
######################


class DialogDataset(Dataset):
    def __init__(
        self, cg: Craigslist, tokenizer: PreTrainedTokenizer, block_size: int = 256
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
    tokenizer: PreTrainedTokenizer,
    train_fp: str,
    val_fp: str,
) -> Tuple[DialogDataset, DialogDataset]:
    # make data
    cg_train = Craigslist(train_fp)
    cg_val = Craigslist(val_fp)

    # train and test
    dd = lambda d: DialogDataset(d, tokenizer=tokenizer)
    return dd(cg_train), dd(cg_val)


class LoraTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs.pop("token_type_ids")
        return super().compute_loss(model, inputs, return_outputs)

###############
# MAIN SCRIPT #
###############
def train(args):
    # make tokenizer
    token = AutoTokenizer.from_pretrained(args.model_name)
    token.add_tokens(["$PRICE", "$PARTNER_PRICE", "<sep>"])
    # token.add_special_token({"mask_token":"[MASK]"})

    # make dataset
    train_dataset, val_dataset = make_craigslist_dataset(
        tokenizer=token, train_fp=args.train_fp, val_fp=args.val_fp
    )

    # initialize model
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(token))
    
    lora_config = LoraConfig(
        r=args.lora_r,
        target_modules=["query_key_value"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        bias="all",
    )
    # PeftModelForCausalLM.from_pretrained(model, )
    model:PeftModelForCausalLM = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(model)

    # finetune
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        # fp16=args.fp16,
        no_cuda=False,
        num_train_epochs=args.epoch,
        # max_steps=20,
        save_strategy="epoch",
        # save_steps=500,
        evaluation_strategy="epoch",
        # eval_steps=500,
        save_total_limit=1,
        warmup_steps=100,
        dataloader_drop_last=True,
        load_best_model_at_end=True,
        report_to="wandb",
        push_to_hub=True,
    )
    trainer = LoraTrainer(
        model,
        train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=token,
        data_collator=default_data_collator,
    )

    trainer.train()

    model.push_to_hub(repo_id="tjddn0402/alpaca_lora")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-fp", default="/opt/ml/chai-naacl-2022/data/fixed_final_translated_train.json")
    parser.add_argument("--val-fp", default="/opt/ml/chai-naacl-2022/data/fixed_final_translated_dev.json")

    parser.add_argument("--model-name", default="EleutherAI/polyglot-ko-5.8b")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--output-dir", default="./alpaca_lora")

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    train(args)