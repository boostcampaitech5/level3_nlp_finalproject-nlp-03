import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline,
)
from peft import PeftModelForCausalLM, PeftConfig, prepare_model_for_int8_training
from typing import Union
from chat_bot.neural_chat.conversation import Conversation
from chat_bot.neural_chat.advisor import Advisor


class E2ELoRA(torch.nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        device: Union[torch.device, str],
        do_quantize: bool = True,
        use_adapter: bool = True,
    ):
        super().__init__()
        self.device = device
        peft_config = PeftConfig.from_pretrained(checkpoint_path)
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=bnb_config if do_quantize else None,
        )
        if do_quantize:
            self.model = prepare_model_for_int8_training(self.model)
        if use_adapter:
            self.model = PeftModelForCausalLM.from_pretrained(
                self.model, checkpoint_path
            )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    def generate(self, conv: Conversation, gen_config: GenerationConfig) -> str:
        with torch.no_grad():
            advisor = Advisor(conv)

            # 구매자의 마지막 메세지로부터 prefix를 가져옴
            advice = advisor.get_force_prefix(conv.messages[-2][1])

            # rule-base로 답변이 정해진 경우, generation을 하지 않고 바로 return합니다.
            if advice.endswith(conv.sep2):
                return advice[: -len(conv.sep2)]

            prompt = conv.get_prompt() + advice
            tokens = self.tokenizer(
                prompt,
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
            response = gen_text.strip().split(conv.sep2)[-2].strip()
            response = response.split(f"{conv.roles[1]}:")[-1].strip()
            response = advisor.replace_outrange_seller_price(response)

            del tokens
            del gen_tokens
            torch.cuda.empty_cache()

            return response


if __name__ == "__main__":
    import json
    import random
    from transformers import GenerationConfig
    from chat_bot.neural_chat.conversation import get_default_conv_template

    lora = E2ELoRA(
        "ggul-tiger/kullm-12.8b-negobot-372data",
        "cuda",
    )
    with open("./data/processed_generated_dev.json", "r", encoding="utf-8") as f:
        datas = json.load(f)

    gen_config = GenerationConfig(
        max_new_tokens=128,
        use_cahce=True,
        early_stopping=True,
        do_sample=True,
        top_k=50,
        top_p=0.85,
        num_beams=3,
        temperature=0.9,
    )

    for i in range(3):
        example = random.choice(datas)
        ev_len = random.randint(0, len(example["events"]))
        ev_len -= not ev_len % 2
        conv = get_default_conv_template()
        conv.scenario = {k: example[k] for k in conv.scenario_key_mapping.keys()}
        conv.scenario["seller_bottom_price"] = conv.scenario["price"] // 2
        for j in range(ev_len):
            conv.append_message(
                example["events"][j]["role"], example["events"][j]["message"]
            )
        conv.append_message("판매자", "")
        gen = lora.generate(conv, gen_config)
        print(f"##### example {i + 1} #####")
        print(conv.get_prompt())
        print(gen)
