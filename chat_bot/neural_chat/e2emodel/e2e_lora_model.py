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

        response = gen_text.strip().split(conv.sep2)[-2].strip()
        response = response.split(f"{conv.roles[1]}:")[-1].strip()

        return response


if __name__ == "__main__":
    lora = E2ELoRA(
        "/opt/ml/level3/chat_bot/logs/checkpoint-87",
        "cuda",
    )

    # examples = [
    #     "제목: 듀얼 출력 마이크로 USB 및 LED 조명이 있는 Verizon 차량용 충전기<|sep|>상품 설명: 이동 중에 두 장치를 동시에 충전하십시오. 추가 USB 포트가 있는 이 차량용 충전기는 한 번에 두 개의 장치를 충전하기에 충분한 전력을 제공합니다. 푸시 버튼 활성화 LED 커넥터 표시등은 더 이상 어둠 속에서 장치를 연결하려고 시도하는 것을 의미합니다. 자동 감지 IC 기술은 장치 유형과 향상된 호환성을 위해 특정 충전 요구 사항을 자동으로 감지합니다. 그리고 내장된 표시등이 빨간색으로 켜져 충전기에 전원이 공급되고 있고 전원 소켓이 제대로 작동하고 있음을 알려줍니다.<|sep|>가격: 10<|sep|>구매자: 안녕하세요, 충전기가 제 차에 맞는지 잘 모르겠습니다. 5달러에 팔 수 있나요?<|sep|>판매자: ",
    #     "제목: 솔 공화국 무선 블루투스 헤드폰<|sep|>상품 설명: 선명한 고음과 섬세하고 깊은 저음을 제공하는 A2 사운드 엔진\n초강력 무선 - 최대 150피트 떨어진 곳에서 공기를 추적하는 제어\n휴대폰, 태블릿 또는 컴퓨터를 포함하여 두 장치에 동시에 연결합니다. 태블릿에서 영화를 보면서 걸려오는 전화에 원활하게 응답하세요.\n전화 통화용 내장 마이크는 핸즈프리 통화를 가능하게 합니다.<|sep|>가격: 155<|sep|>구매자: 헤드폰을 사용하셨나요?<|sep|>판매자: 아니요, 새것이고 아주 좋습니다.<|sep|>판매자: offer 125.0<|sep|>구매자: 게임 콘솔과 함께 사용해도 되나요?<|sep|>판매자: ",
    #     "제목: BLU R1 HD - 16GB<|sep|>상품 설명: BLU R1 HD - 16GB - 블랙 - 아마존 프라임 독점 - 잠금 화면 제공 및 광고 포함. 케이스와 함께 제공됩니다.<|sep|>가격: 50<|sep|>구매자: 안녕하세요?<|sep|>판매자: 네, 잘 지내세요? 제 Blu RD H1 휴대폰에 관심이 있으신 것 같네요<|sep|>구매자: 네, 잘 모르겠어요. 16<|sep|>판매자: 16GB의 내부 저장 공간이 있고 최대 32GB까지 확장할 수 있는 멋진 휴대폰입니다. 또한 2GB 램과 멋진 3GHz 쿼드 코어 프로세서가 탑재되어 있어 생산성과 게임 플레이를 완벽하게 지원합니다.<|sep|>구매자: 휴대폰은 얼마나 오래 되었나요?<|sep|>판매자: 1년 전에 나왔지만 지금 판매하는 휴대폰은 3개월밖에 안 됐어요. 업무 때문에 안드로이드 대신 아이폰으로 바꾸기로 결정해서 판매합니다. 휴대폰에 케이스가 함께 들어있는데, 개통할 때부터 붙어있던 케이스입니다<|sep|>구매자: 케이스는 어떤 색인가요?<|sep|>판매자: 검은색과 파란색, 하드 쉘과 추가 보호를 위해 모서리가 올라간 실리콘 케이스<|sep|>구매자: 더 저렴하게 구입해야 하는데 어떻게 하면 최선인가요?<|sep|>판매자: ",
    #     "제목: 2011 MINI 쿠퍼 하드탑<|sep|>상품 설명: 블랙 스트립이 있는 아이스 블루. 블랙 인조가죽 시트, 16인치 블랙 알로이 휠, 6-스타 스포크. Bluetooth 및 USB/iPad 어댑터. STEPTRONIC 자동 변속기. 모든 정기 유지보수가 완료되었으며 서류 작업이 포함됩니다.<|sep|>가격: 10200<|sep|>판매자: 안녕하세요! 제 차 구매에 관심이 있으신가요?<|sep|>구매자: 네, 그렇습니다. 겉보기에는 멋져 보이는데 주행거리가 얼마나 되나요?<|sep|>판매자: 7만 5천 마일입니다. 모든 정비를 마쳤고 서류도 있습니다.<|sep|>구매자: 더 적은 마일을 원했는데요. 하지만 주행거리가 그렇게 많으니 6천 달러에 팔면 어떨까요?<|sep|>판매자: 그 가격은 저에게는 너무 낮습니다. 9000달러에 팔 수 있을 것 같습니다.<|sep|>구매자: 그래도 너무 비싸네요. 아직 보증이 남아 있나요?<|sep|>판매자: 네, 7년 보증이 아직 1년 남았습니다. 8750달러까지 낮출 수 있을 것 같습니다.<|sep|>구매자: 좋은 가격이네요. 8750달러<|sep|>구매자: offer 8750.0<|sep|>판매자: ",
    # ]
    with open('/opt/ml/level3/data/270_generation_sample.json', 'r') as f:
        examples = json.load(f)

    for idx, example in enumerate(examples):
        gen = lora.generate(example)
        print(f"##### example {idx + 1} #####")
        # print(example.replace("<|sep|>", "\n"))
        print(gen)
