import dataclasses
from typing import List, Dict, Optional


def get_default_conv_template():
    """End to End 모델을 위한 기본 탬플릿입니다."""
    return Conversation(
        system="중고거래 판매자와 구매자의 대화입니다. 판매자는 구매자의 질문에 성실히 답변하고, 판매 가격을 최대화합니다.",
        roles=["구매자", "판매자"],
        scenario={},
        scenario_key_mapping={"title": "제목", "description": "상품 설명", "price": "가격"},
        messages=[],
        sep="\n",
        sep2="<|endoftext|>",
    )


@dataclasses.dataclass
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
    # 모든 메세지의 리스트. [역할, 메세지] 형태로 저장됩니다.
    messages: List[List[str]]
    #
    sep: str
    sep2: str = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        seps = [self.sep, self.sep2]
        ret = self.system + seps[0] + self.get_scenario()
        for i, (role, message) in enumerate(self.messages):
            if message:
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ": "
        return ret

    def get_scenario(self) -> str:
        """시나리오를 문자열로 반환합니다."""
        return (
            self.sep.join(
                [
                    f"[{v}] {self.scenario[k]}"
                    for k, v in self.scenario_key_mapping.items()
                ]
            )
            + self.sep
        )

    def append_message(self, role: str, message: str):
        """새로운 메세지를 추가합니다."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """마지막 메세지를 수정합니다."""
        self.messages[-1][1] = message

    def load_dict(self, formatted_dict: dict):
        """format된 dict를 불러옵니다."""
        self.messages = []
        self.scenario = {k: formatted_dict[k] for k in self.scenario_key_mapping.keys()}
        for i, ev in enumerate(formatted_dict["events"]):
            assert self.roles[i % 2] == ev["role"], "구매자, 판매자 순서로 된 데이터를 입력해주세요."
            self.append_message(ev["role"], ev["message"])


if __name__ == "__main__":
    conv = get_default_conv_template()
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
