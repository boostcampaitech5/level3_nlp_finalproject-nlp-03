import re
from typing import List, Tuple

# 숫자와 매칭됩니다. ex) 34000
NUMBERS = re.compile(r"\d+")

# 만, 천 등의 단위와 결합된 숫자와 매칭됩니다. ex) 3만 4천
NUMBERS_WITH_TEXT = re.compile(r"(?:\d+\s?[만천백]\s?)+")

# 숫자 없이 만, 천 등의 단위로만 구성된 숫자와 매칭됩니다. "원"이 붙어야지만 매칭됩니다. ex)천만원
NO_NUMBER_PRICE = re.compile(r"(?<!\d)(?<!\d\s)(?:[만천백]\s?)+(?=원)")

# 숫자와 결합될 수 있는 금액이 아닌 단위의 모음입니다.
# 안정적으로 금액만 뽑기 위해 아래 단위가 붙으면 금액으로 고려하지 않습니다.
unwanted_units = (
    "mm",
    "cm",
    "m",
    "km",
    "년",
    "월",
    "일",
    "시",
    "분",
    "초",
    "mah",
    "인치",
    "gb",
    "mb",
)

# 1000원만 빼주세요와 같이 할인을 요구하는 상황을 이해하기 위해 사용합니다.
discount_prefixes = ("깎", "빼")


def parse_wanted_price(
    role: str, text: str, seller_wanted_price: int, buyer_wanted_price: int
):
    """
    role을 고려해서 text에서 원하는 금액을 추출합니다.
    금액을 찾지 못하면 -1을 반환합니다.
    ex1) 구매자: 만원은 너무 비싼데 9천원에 팔아주세요. -> 9000
    ex2) 구매자: 안녕하세요. 7월이라 덥네요. -> -1
    """
    if role == "구매자":
        price_list, _ = parse_prices(text, buyer_wanted_price, 0.5, 2)
        if not price_list:
            return -1
        min_price = min(price_list)
        if min_price >= seller_wanted_price:
            return -1  # ex: 5만원은 너무 비싸요.
        return min_price
    if role == "판매자":
        price_list, _ = parse_prices(text, seller_wanted_price, 0.5, 2)
        if not price_list:
            return -1
        max_price = max(price_list)
        if max_price <= buyer_wanted_price:  # ex: 5만원에 판매할 순 없습니다.
            return -1
        return max_price


def parse_prices(
    text: str, ref_price: int, bottom_ratio: float, ceil_ratio: float
) -> Tuple[List[int], List[re.Match]]:
    """
    텍스트에서 금액으로 추정되는 숫자의 리스트를 추출합니다.
    숫자가 ref_price * bottom_ratio보다 작거나 ref_price * ceil_ratio보다 크다면,
    금액을 말하는 것이 아니라고 간주합니다.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(",", "")
    text = text.lower()
    matches = []

    for match in NUMBERS.finditer(text):
        if text[match.end() :].lstrip().startswith(unwanted_units):
            continue  # 금액이 아닌 단위가 붙은 숫자라면 고려하지 않음.
        matches.append(match)
    for match in NUMBERS_WITH_TEXT.finditer(text):
        if text[match.end() :].lstrip().startswith(unwanted_units):
            continue
        matches.append(match)
    for match in NO_NUMBER_PRICE.finditer(text):
        matches.append(match)

    prices = [price_to_int(match.group()) for match in matches]
    final_prices = []
    final_matches = []
    for price, match in zip(prices, matches):
        if price < ref_price * bottom_ratio and any_string_in(["깎", "빼"], text):
            # 원래 가격이 20000원인데, 1000원만 깎아달라고 하면 19000원을 의도한 것으로 간주합니다.
            final_prices.append(ref_price - price)
            final_matches.append(match)
        elif price < ref_price * bottom_ratio or price > ref_price * ceil_ratio:
            continue
        else:
            final_prices.append(price)
            final_matches.append(match)

    return final_prices, final_matches


def price_to_int(price: str) -> int:
    """
    3만 5천과 같이 숫자와 단위가 결합된 경우, 35000으로 원래 숫자를 반환합니다.
    """
    price = price.strip()
    price = re.sub(r"(?<!\d)0", "", price)
    if not price:
        return 0
    price = price.replace("만", "* 10000 +")
    price = price.replace("천", "* 1000 +")
    price = price.replace("백", "* 100 +")

    price = re.sub(r"\+\s*\*", "*", price)
    if price[0] == "*":
        price = price[1:]
    if price[-1] == "+":
        price = price[:-1]

    return eval(price)


def any_string_in(strings: list[str], text: str) -> bool:
    return any([text.find(string) != -1 for string in strings])


if __name__ == "__main__":
    import random
    import json

    with open("./data/processed_generated_train.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for i in range(3):
        d = random.choice(data)
        listed_price = d["price"]
        print(f"sample: {i + 1}")
        for ev in d["events"]:
            print(f"utterance: {ev['role']}: {ev['message']}")
            print(
                f"parsed price: {parse_wanted_price(ev['role'], ev['message'], listed_price, listed_price * 0.8)}"
            )
