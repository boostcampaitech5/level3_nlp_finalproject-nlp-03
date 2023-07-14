import dataclasses
from typing import List, Dict


def get_default_conv_template():
    """End to End 모델을 위한 기본 탬플릿입니다."""
    return Conversation(
        system="중고거래 판매자와 구매자의 대화입니다. 판매자는 구매자의 질문에 성실히 답변하고, 판매 가격을 최대화합니다.",
        roles=["구매자", "판매자"],
        scenario={},
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
            self.sep.join([f"[{k}] {v}" for k, v in self.scenario.items()]) + self.sep
        )

    def append_message(self, role: str, message: str):
        """새로운 메세지를 추가합니다."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """마지막 메세지를 수정합니다."""
        self.messages[-1][1] = message


if __name__ == "__main__":
    conv = get_default_conv_template()
    conv.scenario = {"제목": "아이폰 팔아요", "상품 설명": "지구 최강 아이폰", "가격": 10000}
    conv.append_message(role="구매자", message="안녕하세요!")
    conv.append_message(role="판매자", message="방가방가")
    print(conv.get_prompt())
