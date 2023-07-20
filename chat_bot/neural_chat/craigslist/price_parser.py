import re
from typing import List, Tuple

# 숫자와 매칭됩니다. ex) 34000, 34,000
NUMBERS = re.compile(r"(?=\d+)[\d,]*")

# 만, 천 등의 단위와 결합된 숫자와 매칭됩니다. ex) 3만 4천
NUMBERS_WITH_TEXT = re.compile(r"(?:(?=\d+)[\d,이삼사오육칠팔구]*\s*[억만천백십]+\s*)+")

# # 삼천만(원), 이만(원), 등 텍스트로만 이루어진 숫자 매칭합니다.
# ONE2THOUSAND=r"[[이삼사오육칠팔구]?[천백십]?]*"
# TEXT_EXPRESSION_NUMBER = re.compile(fr"[{ONE2THOUSAND}억]?\s+?[{ONE2THOUSAND}만]?\s+?[{ONE2THOUSAND}?]?원?")

# 숫자 없이 만, 천 등의 단위로만 구성된 숫자와 매칭됩니다. "원"이 붙어야지만 매칭됩니다. ex)천만원
NO_NUMBER_PRICE = re.compile(r"(?<!\d)(?<!\d\s)(?:[이삼사오육칠팔구]?[억만천백십]\s?)+(?=원)")

# 숫자와 결합될 수 있는 금액이 아닌 단위의 모음입니다.
# 안정적으로 금액만 뽑기 위해 아래 단위가 붙으면 금액으로 고려하지 않습니다.
# fmt: off
unwanted_units = (
    "년", "월", "개월", "일", "시", "분", "초",  # 시간
    "테라", "기가", "메가", "헥토", "키로", "킬로", "센티", "센치", "데시", "밀리", "미리", "마이크로", "나노",  # 단위
    "번", "회",  # 횟수
    "개", "매", "송이", "그루",  # 갯수
    "파운드", "온스", "그램", "그람", "되", "홉", "톤",  # 무게
    "동", "호", "층",  # 주소
    "인치", "피트", "마일", "미터",  # 거리
    "평", "평방", "헥타르", "에이커",  # 너비
    "리터", "배럴", "갤런", "쿼트", "파인트",  # 부피
    "파스칼", "토르",  # 압력
    "제곱", "세제곱", 
    "짝", "쪽", 
    "코어", 
    "점", 
    "마력", "기통", "륜",
    # "근", 
    # "장", 
    # "퍼센트", "퍼", "프로", "%" # 깎아주세요, 할인해주세요, 빼주세요와 함께 활용 가능
)
# fmt :on

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
        if re.match(r"##<\d+>##", text):
            return int(text[3:-3])
        price_list, _ = parse_prices(text, buyer_wanted_price, 0.3, 2)
        if not price_list:
            return -1
        min_price = min(price_list)
        if min_price >= seller_wanted_price:
            return -1  # ex: 5만원은 너무 비싸요.
        return min_price
    if role == "판매자":
        price_list, _ = parse_prices(text, seller_wanted_price, 0.3, 2)
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
    text = text.lower()
    matches = []
    eng_greek_letter = re.compile(r"[a-zA-Z\u0370-\u03FF].*$") # 영어 알파벳 & 그리스문자 (모든 단위 배제)

    for match in NUMBERS.finditer(text):
        if eng_greek_letter.match(text[match.end() :].lstrip()):
            continue # 영어는 무조건 단위로 간주함.
        elif text[match.end() :].lstrip().startswith(unwanted_units):
            continue # 금액이 아닌 단위가 붙은 숫자라면 고려하지 않음.
        matches.append(match)
    for match in NUMBERS_WITH_TEXT.finditer(text):
        if eng_greek_letter.match(text[match.end() :].lstrip()):
            continue # 영어는 무조건 단위로 간주함.
        elif text[match.end() :].lstrip().startswith(unwanted_units):
            continue # 금액이 아닌 단위가 붙은 숫자라면 고려하지 않음.
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
    
    sorted_results = sorted(zip(final_prices, final_matches), key=lambda x: x[1].start())
    final_prices = [item[0] for item in sorted_results]
    final_matches = [item[1] for item in sorted_results]

    return final_prices, final_matches


def price_to_int(price: str) -> int:
    """
    3만 5천과 같이 숫자와 단위가 결합된 경우, 35000으로 원래 숫자를 반환합니다.
    """
    price = price.strip()
    price = price.replace(",", "")
    price = price.replace(" ", "")
    price = re.sub(r"(?<!\d)0", "", price)
    if not price:
        return 0
    price = price.replace("일", "1")
    price = price.replace("이", "2")
    price = price.replace("삼", "3")
    price = price.replace("사", "4")
    price = price.replace("오", "5")
    price = price.replace("육", "6")
    price = price.replace("칠", "7")
    price = price.replace("팔", "8")
    price = price.replace("구", "9")

    price = price.replace("천", "* 1000 +")
    price = price.replace("백", "* 100 +")
    price = price.replace("십", "* 10 +")
    price = price.replace("만", "* 10000 +")

    price = re.sub(r"\+\s*\*", "*", price)
    if price[0] == "*":
        price = price[1:]
    if price[-1] == "+":
        price = price[:-1]

    return int(eval(price))


def any_string_in(strings: List[str], text: str) -> bool:
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
