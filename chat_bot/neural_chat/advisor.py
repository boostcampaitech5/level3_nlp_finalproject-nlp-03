from .conversation import Conversation
from .craigslist.price_parser import parse_wanted_price, parse_prices
import re
import random


class Advisor:
    """
    rule-base로 가격을 추적하면서 모델의 출력을 조절하는 클래스입니다.
    """

    def __init__(self, conv: Conversation):
        """
        initialize될 때, 자동으로 모든 conversation을 읽어서 구매자와 판매자의 희망 가격을 찾습니다.
        """
        self.eos_token = conv.sep2
        if isinstance(conv.scenario["price"], str):
            self.listed_price = int(re.sub(r"[^\d]+", "", conv.scenario["price"]))
        else:
            self.listed_price = int(conv.scenario["price"])
        self.desired_price = dict()
        self.desired_price["판매자"] = self.listed_price
        self.seller_bottom_price = (
            conv.scenario["seller_bottom_price"]
            if "seller_bottom_price" in conv.scenario.keys()
            else int(0.4 * self.listed_price)
            # scenario에 bottom_price가 없는 경우 임의의 값으로 설정합니다.
        )
        self.desired_price["구매자"] = self.seller_bottom_price
        self.last_nego_agent = "판매자"

        for role, message in conv.messages:
            self.update_price(role, message)

    def get_advice(self, message: str) -> str:
        """
        Legacy입니다. 현재는 사용하지 않습니다.
        제 3의 발화자인 조언자가 모델에게 조언을 합니다.
        """
        if re.match(r"##<\d+>##", message):
            return "조언자: 구매자가 최종 가격을 제시했습니다. 신중하게 고민해서 ##<수락>##이나 ##<거절>##로 답변하세요."
        price = parse_wanted_price(
            "구매자", message, self.desired_price["판매자"], self.desired_price["구매자"]
        )
        if price != -1:
            self.last_nego_agent = "구매자"
        if price != -1 and price < self.seller_bottom_price:
            return f'조언자: """구매자가 너무 낮은 금액을 제시했습니다. 절대 수락하지 마세요."""'
        elif price == -1:
            return '조언자: """친절하고 자세한 답변을 해주세요."""'
        elif price == self.desired_price["판매자"]:
            return '조언자: """구매자가 판매자가 제시한 가격에 동의했습니다. 감사 인사를 전하세요."""'
        elif price > self.desired_price["구매자"]:
            return '조언자: """구매자가 희망 금액을 높였습니다. 수락하거나 더 적절한 가격을 제시하세요."""'
        elif price == self.desired_price["구매자"]:
            return '조언자: """구매자가 이전과 같은 금액을 요구하고 있습니다. 수락하거나 더 적극적으로 구매자를 설득해주세요."""'
        elif price < self.desired_price["구매자"]:
            return '조언자: """구매자가 희망 금액을 더 낮췄습니다. 절대 수락하지 마시고, 이전에 합의된 금액을 알려주세요."""'

    def get_force_prefix(self, buyer_message: str) -> str:
        """
        구매자의 발화로부터 모델이 반드시 생성해야할 prefix를 return합니다.
        """
        price = parse_wanted_price(
            "구매자", buyer_message, self.desired_price["판매자"], self.desired_price["구매자"]
        )

        # 최종 가격 제안이 들어온 경우
        if re.match(r"##<\d+>##", buyer_message):
            # 마지막으로 합의된 가격보다 낮은 가격이 제시된 경우
            if price < self.desired_price[self.last_nego_agent]:
                return f"##<거절>##{self.eos_token}"
            # 판매자가 원했던 가격에 제안이 들어온 경우.
            elif price >= self.desired_price["판매자"]:
                return f"##<수락>##{self.eos_token}"
            # 가격 합의가 이루어지지 않았을 때 제안이 들어온 경우. 모델에게 판단을 맡김.
            else:
                return "##<"

        # 가격에 대한 얘기가 나오지 않은 경우.
        if price == -1:
            return ""

        # 판매자 최저 판매 금액보다 낮은 금액이 제시된 경우.
        if price < self.seller_bottom_price:
            return random.choice(
                [
                    f"죄송하지만, 제가 생각한 가격보다 너무 낮네요. 제가 설정한 가격이 {self.listed_price}원 인 것을 감안해주세요.",
                    f"그렇게 낮은 가격에 드리기는 어려워요. 제가 설정한 가격이 {self.listed_price}원입니다.",
                    f"제시하신 가격이 너무 낮아요. 제가 설정한 가격이 {self.listed_price}원 인 것을 감안해주세요.",
                    f"제가 설정한 가격이 {self.listed_price}원 인 것을 감안하면, {price}원은 너무 낮아요.",
                    f"죄송하지만 {price}원에 드리긴 어렵네요. 제가 설정한 가격이 {self.listed_price}원 인 것을 감안해주세요.",
                ]
            )
        # 구매자가 이전에 제시했던 금액보다 더 낮은 금액을 제시한 경우.
        elif price < self.desired_price["구매자"]:
            return random.choice(
                [
                    "죄송하지만 더 깎아드릴 순 없어요.",
                    "가격을 더 내려드리진 못해요.",
                    "이전에 합의한 금액보다 더 낮은 금액은 안 돼요.",
                    "죄송하지만, 더 깎아드리긴 어려워요.",
                    "이전보다 낮은 금액에 판매하긴 어려워요.",
                ]
            )

        return ""

    def replace_outrange_seller_price(self, message: str) -> str:
        """판매자의 발화에서 엉뚱한 금액이 나온 경우 강제로 금액을 바꿉니다."""
        prices, matches = parse_prices(message, self.desired_price["판매자"], 0.5, 1.5)

        if not prices:
            return message

        for price, match in zip(reversed(prices), reversed(matches)):
            # 합의된 가격보다 높은 가격을 제시한 경우
            if price > self.desired_price["판매자"]:
                message = (
                    message[: match.start()]
                    + str(self.desired_price["판매자"])
                    + "원"
                    + message[match.end() :]
                )

        return message

    def update_price(self, role: str, message: str):
        """구매자와 판매자의 희망 가격을 업데이트합니다."""
        if re.match(r"##<\d+>##", message):
            return

        price = parse_wanted_price(
            role, message, self.desired_price["판매자"], self.desired_price["구매자"]
        )

        if role == "구매자":
            if price != -1:
                self.last_nego_agent = "구매자"
                self.desired_price["구매자"] = max(price, self.desired_price["구매자"])
        elif role == "판매자":
            if price != -1:
                self.last_nego_agent = "판매자"
                self.desired_price["판매자"] = min(price, self.desired_price["판매자"])
