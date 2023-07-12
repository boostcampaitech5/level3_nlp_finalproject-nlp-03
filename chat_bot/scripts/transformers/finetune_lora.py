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
    get_peft_model,
    prepare_model_for_int8_training,
)
from chat_bot.neural_chat.craigslist.e2e_dataset import (
    SimpleDialogDataset,
    VicunaDialogDataset,
)
import argparse


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
            args.train_fp, tokenizer=tokenizer, block_size=256
        )
    elif args.dataset_type == "vicuna":
        train_dataset = VicunaDialogDataset(args.train_fp, tokenizer=tokenizer)
    else:
        raise NotImplementedError

    # QLoRA configs
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        target_modules=["query_key_value"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
        inference_mode=False,
        bias="all",
    )

    # initialize model
    print("load model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config
    )

    print("get peft model...")
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, lora_config)
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
        run_name=args.run_name,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-fp",
        default="/opt/ml/level3_nlp_finalproject-nlp-03/data/chatbot_train.json",
    )

    parser.add_argument("--model-name", default="nlpai-lab/kullm-polyglot-12.8b-v2")
    parser.add_argument("--dataset-type", default="vicuna")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="./chat_bot/logs/kullm-12.8b")

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--run-name", default="QLoRA")
    args = parser.parse_args()

    train(args)
