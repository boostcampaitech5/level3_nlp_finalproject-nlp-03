import dataclasses
from typing import List, Dict, Optional, Union
from transformers import PreTrainedTokenizer, AutoTokenizer
from dataclasses import dataclass, field
import re
import sys
sys.path.append("./")
from chat_bot.neural_chat.craigslist.price_parser import parse_prices, num2won


def get_conv_template(template_name: str):
    """
    원하는 conversation template을 가져옵니다.
    default, v2, price_weak, simple_weak를 지원합니다.
    """
    return CONV_TEMPLATES[template_name]()


def get_default_conv_template():
    """End to End 모델 훈련을 위한 기본 탬플릿입니다."""
    return Conversation(
        system="중고거래 판매자와 구매자의 대화입니다. 판매자는 구매자의 질문에 성실히 답변하고, 판매 가격을 최대화합니다.",
        roles=["구매자", "판매자"],
        scenario={},
        scenario_key_mapping={"title": "제목", "description": "상품 설명", "price": "가격"},
        scenario_format="bracket",
        messages=[],
        sep="\n",
        sep2="<|endoftext|>",
    )


def get_v2_conv_template():
    """End to End 모델 훈련을 위한 버전 2 탬플릿입니다."""
    return Conversation(
        system="당신은 중고거래 판매자입니다. 판매자는 상품 가격과 구매자와 판매자가 제시한 가격을 참고해서 적절한 근거와 함께 합의 가격을 제시합니다.",
        roles=["구매자", "판매자"],
        scenario={},
        scenario_key_mapping={"title": "제목", "description": "상품 설명", "price": "상품 가격"},
        scenario_format="colon",
        messages=[],
        sep="\n",
        sep2="<|endoftext|>",
    )


def get_price_weak_conv_template():
    """가격과 관련된 weak case 훈련을 위한 탬플릿입니다."""
    return Conversation(
        system="당신은 중고거래 판매자입니다. 판매자는 상품 가격과 구매자와 판매자가 제시한 가격을 참고해서 적절한 근거와 함께 합의 가격을 제시합니다.",
        roles=["구매자", "판매자"],
        scenario={},
        scenario_key_mapping={"price": "상품 가격"},
        scenario_format="colon",
        messages=[],
        sep="\n",
        sep2="<|endoftext|>",
    )


def get_simple_weak_conv_template():
    """가격과 무관한 weak case 훈련을 위한 탬플릿입니다."""
    return Conversation(
        system="",
        roles=["구매자", "판매자"],
        scenario={},
        scenario_key_mapping={},
        scenario_format="colon",
        messages=[],
        sep="\n",
        sep2="<|endoftext|>",
    )


CONV_TEMPLATES = {
    "default": get_default_conv_template,
    "v2": get_v2_conv_template,
    "price_weak": get_price_weak_conv_template,
    "simple_weak": get_simple_weak_conv_template,
}


@dataclass
class Conversation:
    """프롬프트, 상품 판매 정보, 채팅 내역을 저장하는 클래스입니다."""

    # 시스템 프롬프트
    system: str
    # 두 역할 이름
    roles: List[str]
    # 제목, 상품 설명, 가격 등의 정보가 저장되는 시나리오 정보
    scenario: Dict
    # 프롬프트를 생성할 때 scenario에 있는 title, dict 등의 key를 어떤 단어로 mapping할지
    scenario_key_mapping: Dict
    # 시나리오의 포맷을 결정합니다. 예시) 1. bracket - [제목] 아이폰 팔아요 2. colon 제목: 아이폰 팔아요
    scenario_format: str
    # 모든 메세지의 리스트. [역할, 메세지] 형태로 저장됩니다.
    messages: List[List[str]]
    # seperater
    sep: str
    sep2: str

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        "nlpai-lab/kullm-polyglot-12.8b-v2"
    )
    max_token: int = 1024

    hangeul_price: bool = False
    desired_price: Dict = field(default_factory=dict)

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        accum_token_num = 0
        seps = [self.sep, self.sep2]

        scenario = self.system + seps[0] + self.get_scenario()
        accum_token_num += len(self.tokenizer(scenario)["input_ids"])

        dialog = []
        for i, (role, message) in enumerate(self.messages):
            if message:
                append_msg = role + ": " + message + seps[i % 2]
            else:
                append_msg = role + ": "
            dialog.append(append_msg)

        for i, msg in enumerate(reversed(dialog), 1):
            accum_token_num += len(self.tokenizer(msg)["input_ids"])
            if accum_token_num > self.max_token:
                dialog = dialog[-i:]
                break

        dialog = "".join(dialog)
        return scenario + dialog

    def get_scenario(self) -> str:
        """시나리오를 문자열로 반환합니다."""
        if self.scenario_format == "bracket":
            info_list = [
                f"[{v}] {num2won(self.scenario[k])}"
                if k == "price" and self.hangeul_price else
                f"[{v}] {self.scenario[k]}원"
                if k == "price" else
                f"[{v}] {self.scenario[k]}"
                for k, v in self.scenario_key_mapping.items()
            ]
        elif self.scenario_format == "colon":
            info_list = [
                f"{v}: {num2won(self.scenario[k])}"
                if k == "price" and self.hangeul_price else
                f"{v}: {self.scenario[k]}원"
                if k == "price" else
                f"{v}: {self.scenario[k]}"
                for k, v in self.scenario_key_mapping.items()
            ]
        return self.sep.join(info_list) + self.sep + self.sep

    def append_message(self, role: str, message: str):
        """새로운 메세지를 추가합니다."""
        if self.hangeul_price:
            try:
                if role == "구매자" and re.match(r"##<\d+>##", message):
                        message="##<"+num2won(int(message[3:-3]))+">##"
                else:
                    matched_prices, price_matches = parse_prices(message, self.desired_price[role], 0., 999999999)
                    for price, match in zip(reversed(matched_prices),reversed(price_matches)):
                        message=message[:match.start()]+num2won(price)+message[match.end():]
            except Exception as e:
                print(e)
                
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """마지막 메세지를 수정합니다."""
        self.messages[-1][1] = message

    def load_dict(self, formatted_dict: dict):
        """format된 dict를 불러옵니다."""
        self.messages = []
        self.scenario = {k: formatted_dict[k] for k in self.scenario_key_mapping.keys()}
        self.desired_price["판매자"] = self.scenario["price"]
        self.desired_price["구매자"] = self.scenario["price"]*0.8
        for i, ev in enumerate(formatted_dict["events"]):
            assert self.roles[i % 2] == ev["role"], "구매자, 판매자 순서로 된 데이터를 입력해주세요."
            self.append_message(ev["role"], ev["message"])


if __name__ == "__main__":
    conv = get_conv_template("v2")
    sample_data = {
        "title": "아이폰 팔아요",
        "description": "지구 최강 아이폰",
        "price": 10000,
        "events": [
            {"role": "구매자", "message": "안녕하세요!"},
            {"role": "판매자", "message": "방가방가"},
        ],
    }
    conv.load_dict(sample_data)
    print(conv.get_prompt())
