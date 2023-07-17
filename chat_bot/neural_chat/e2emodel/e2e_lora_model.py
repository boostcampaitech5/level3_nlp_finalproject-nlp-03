import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModelForCausalLM, PeftConfig, prepare_model_for_int8_training
from typing import Union
from chat_bot.neural_chat.conversation import Conversation
import json, re 
from chat_bot.neural_chat.conversation import get_default_conv_template

def convert_to_model_input(example):
    conv = get_default_conv_template()
    conv.scenario["제목"] = example['title']
    conv.scenario["상품 설명"] = example['description']
    conv.scenario["가격"] = example['price']
    conv.messages = example['events']
    conv.append_message(conv.roles[1], "")
    return conv
    
class E2ELoRA(torch.nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        device: Union[torch.device, str],
        do_quantize: bool = True,
    ):
        super().__init__()
        self.device = device
        peft_config = PeftConfig.from_pretrained(checkpoint_path)
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=bnb_config if do_quantize else None,
        )
        if do_quantize:
            model = prepare_model_for_int8_training(model)
        model = PeftModelForCausalLM.from_pretrained(model, checkpoint_path)
        self.model = model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    def generate(self, conv: Conversation, gen_config: GenerationConfig) -> str:
        tokens = self.tokenizer(
            conv.get_prompt(),
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.device)

        gen_tokens = self.model.generate(
            **tokens,
            generation_config=gen_config,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        gen_text = self.tokenizer.decode(gen_tokens[0])

        del tokens
        del gen_tokens
        torch.cuda.empty_cache()
        # import re 
        # response = re.findall(r"판매자:(.+)<\|endoftext\|>$", gen_text)[-1]
        response = gen_text.strip().split(conv.sep2)[-2].strip()
        response = response.split(f"{conv.roles[1]}:")[-1].strip()
        return response.strip()


if __name__ == "__main__":
    lora = E2ELoRA(
        "/opt/ml/level3/chat_bot/logs/checkpoint-87",
        "cuda",
    )
    
    gen_config = GenerationConfig(
        max_new_tokens=128,
        use_cahce=False,
        early_stopping=True,
        do_sample=True,
        top_k=100,
        top_p=0.85,
        num_beams=5,
        temperature=0.7,
    )
    
    with open('/opt/ml/level3/data/270_generation_sample.json', 'r') as f:
        examples = json.load(f)

    for idx, example in enumerate(examples):
        gen = lora.generate(convert_to_model_input(example), gen_config)
        print(f"##### example {idx + 1} #####")
        # print(example.replace("<|sep|>", "\n"))
        print(gen)
