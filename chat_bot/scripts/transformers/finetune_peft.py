import sys
sys.path.append("./")

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
from chat_bot.neural_chat.craigslist.e2e_dataset import (
    SimpleDialogDataset,
    VicunaDialogDataset,
)
import argparse
import json


###############
# MAIN SCRIPT #
###############
def train(args):
    # make tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = "<|sep|>"
    tokenizer.model_max_length = args.max_length

    # make dataset
    if args.dataset_type == "simple":
        train_dataset = SimpleDialogDataset(
            args.train_fp, tokenizer=tokenizer, split="train", block_size=256
        )
    elif args.dataset_type == "vicuna":
        train_dataset = VicunaDialogDataset(args.train_fp, tokenizer=tokenizer, split="train")
    else:
        raise NotImplementedError

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
        report_to="wandb",
        # push_to_hub=True,
        run_name=args.run_name,
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
        default="ggul-tiger/negobot_cleaned_100",
    )

    parser.add_argument("--model-name", default="nlpai-lab/kullm-polyglot-12.8b-v2")
    parser.add_argument("--dataset-type", default="vicuna")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="./chat_bot/logs/kullm-polyglot-12.8b-prefix")
    parser.add_argument("--run-name", default="prefix_tuning")

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
