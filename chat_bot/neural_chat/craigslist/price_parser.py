import re
from typing import List, Tuple

# 10000보다 작은 수까지 match
UNDER_10K = re.compile(
    r"(([\.,\d]*|[일이삼사오육칠팔구])(천|처넌)\s*)?(([\.,\d]*|[일이삼사오육칠팔구])백\s*)?(([\.,\d]*|[일이삼사오육칠팔구])십\s*)?(([\.,\d]*|[일이삼사오육칠팔구])\s*)?"
)
# 억단위까지 match
MONEY_TEXT = re.compile(
    r"((?<=^)|(?<=[\s\t\r\f\n\v]))(?=([\d일이삼사오육칠팔구십백천만억₩][\.,\d일이삼사오육칠팔구십백천만억원처마]|처넌|마넌))(₩\s*)?("
    + UNDER_10K.pattern
    + r"억)?\s*("
    + UNDER_10K.pattern
    + r"(만|마넌))?\s*"
    + UNDER_10K.pattern
    + r"[원냥₩]?"
)

# 숫자와 결합될 수 있는 금액이 아닌 단위의 모음입니다.
# 안정적으로 금액만 뽑기 위해 아래 단위가 붙으면 금액으로 고려하지 않습니다.
# fmt: off
NOT_MONEY_UNITS = (
    "년", "월", "개월", "일", "시", "분", "초",  # 시간
    "테라", "기가", "메가", "헥토", "키로", "킬로", "센티", "센치", "데시", "밀리", "미리", "마이크로", "나노",  # 단위
    "번", "회",  # 횟수
    "개", "매", "송이", "그루", "명", "병",  # 갯수
    "파운드", "온스", "그램", "그람", "되", "홉", "톤",  # 무게
    "동", "호", "층",  # 주소
    "인치", "피트", "마일", "미터",  # 거리
    "평", "평방", "헥타르", "에이커",  # 너비
    "리터", "배럴", "갤런", "쿼트", "파인트",  # 부피
    "파스칼", "토르",  # 압력
    "세", "살", # 나이
    "퍼", "프로", "%", "/", "할", "푼", "리", # 비율
    "제곱", "세제곱", 
    "짝", "쪽",                                       
    "코어", 
    "토큰",
    "점", 
    "마력", "기통", "륜",
    # "근", 
    # "장", 
    # "퍼센트", "퍼", "프로", "%" # 깎아주세요, 할인해주세요, 빼주세요와 함께 활용 가능
)
# fmt :on
ENG_GREEK_LETTERS = re.compile(r"[a-zA-Z\u0370-\u03FF].*$") # 영어 알파벳 & 그리스문자 (모든 단위 배제)
SEARCH_DISCOUNT_WORD_UPTO=10

# 비율을 나타내는 표현
PERCENTAGE = re.compile(r"\d{1,3}\s*(퍼(센트)?|%)") # 10프로 <- 아이패드 10프로 가능하므로 제외함.
HALPUNRI = re.compile(r"(?=[\d일이삼사오육칠팔구])([\d일이삼사오육칠팔구]\s*할)?\s*([\d일이삼사오육칠팔구십]\s*푼)?\s*([\d일이삼사오육칠팔구]\s*리)?") # 주의 : '' match 가능하므로 '' 를 return하는 경우 제거해야
FRACTIONAL_OVER = re.compile(r"(?=[\d일이삼사오육칠팔구십])(\d+|[일이삼사오육칠팔구십]+)+\s*분(의|에)\s*(\d+|[일이삼사오육칠팔구십]+)+")
FRACTIONAL = re.compile(r"\d+\s*/\s*\d+")
HALF = re.compile(r"(절반|반값(?!\s*택배))")

def match_ratio(text:str)->Tuple[List[float], List[re.Match]]:
    """1 턴의 발화를 입력받아서 비율을 나타내는 단어를 0-1 사이 실수로 변환"""
    ratios, matches=[], []

    for m in re.finditer(PERCENTAGE, text):
        ratio=float(re.match(r'\d+',m.group()))/100
        if ratio<=1.0:
            matches.append(m)
            ratios.append(ratio)

    n2i = {n:i for i, n in enumerate("일이삼사오육칠팔구", 1)}
    for m in re.finditer(HALPUNRI, text):
        ratio=0
        if '할' in m.group():
            n,m=re.split(r'\s*할\s*',m.group())
            if n in n2i.keys():
                ratio += float(n2i[n])/10
            else:
                ratio += float(n.strip())/10
        if '푼' in m.group():
            n,m=re.split(r'\s*푼\s*',m.group())
            if n in n2i.keys():
                ratio += float(n2i[n])/100
            else:
                ratio += float(n.strip())/100
        if '리' in m.group():
            n,m=re.split(r'\s*리\s*',m.group())
            if n in n2i.keys():
                ratio += float(n2i[n])/1000
            else:
                ratio += float(n.strip())/1000
        if ratio<=1.0:
            matches.append(m)
            ratios.append(ratio)

    for m in re.finditer(FRACTIONAL_OVER, text):
        denom, numer = re.split(r"\s*분(의|에)\s*", m.group())
        denom = str2int_under10k(denom)
        numer = str2int_under10k(numer)
        ratio = float(numer)/float(denom)
        if ratio<=1.0:
            matches.append(m)
            ratios.append(ratio)

    for m in re.finditer(FRACTIONAL, text):
        numer, denom = re.split(r"\s*/\s*", m.group())
        ratio = float(numer)/float(denom)
        if ratio<=1.0:
            matches.append(m)
            ratios.append(ratio)

    for m in re.finditer(HALF, text):
        matches.append(m)
        ratios.append(0.5)
    
    return ratios, matches

# 할인을 암시하는 표현
DISCOUNT = re.compile(r"(에누리|에눌|할인|세일|네고|깎|깍|빼)")
CATCH_DISCOUNT=True

def parse_wanted_price(
    role: str, text: str, seller_wanted_price: int, buyer_wanted_price: int
) -> int:
    """
    role을 고려해서 text에서 원하는 금액을 추출합니다.
    금액을 찾지 못하면 -1을 반환합니다.
    ex1) 구매자: 만원은 너무 비싼데 9천원에 팔아주세요. -> 9000
    ex2) 구매자: 안녕하세요. 7월이라 덥네요. -> -1
    """
    if role == "구매자":
        if re.match(r"##<\d+>##", text):
            return int(text[3:-3])
        price_list, _ = parse_prices(text, seller_wanted_price, 0.3, 2)
        if not price_list:
            return -1
        if len(price_list) == 1 and any_string_in(["비싸", "비싼", "높", "부담"], text):
            return -1
        min_price = min(price_list)
        if min_price > seller_wanted_price:
            return -1
        return min_price
    if role == "판매자":
        price_list, _ = parse_prices(text, buyer_wanted_price, 0.3, 2)
        if not price_list:
            return -1
        if len(price_list) == 1 and any_string_in(["죄송", "낮"], text):
            return -1
        max_price = max(price_list)
        if max_price < buyer_wanted_price:
            return -1
        return max_price


def parse_prices(
    text: str, ref_price: int, bottom_ratio: float, ceil_ratio: float
) -> Tuple[List[int], List[re.Match]]:
    """
    하나의 message에서 금액으로 추정되는 숫자의 리스트를 추출합니다.
    숫자가 ref_price * bottom_ratio보다 작거나 ref_price * ceil_ratio보다 크다면,
    금액을 말하는 것이 아니라고 간주합니다.
    """
    price_matches = list(MONEY_TEXT.finditer(text))
    price_matches.sort(key=lambda match: match.span()) # index 순으로 정렬
    # print(matches)

    # 숫자 match된 문자열에서 가격 아닌거 거르기
    valid_price_matches=[]
    for match in price_matches:
        if re.match(r'^\s*$',match.group()):
            # 공백만 매치되는 케이스 거르기
            continue
        elif ENG_GREEK_LETTERS.match(text[match.end() :].lstrip()):
            continue # 영어는 무조건 단위로 간주함.
        elif not bool(re.search(r"₩|원|냥",match.group())) and text[match.end() :].lstrip().startswith(NOT_MONEY_UNITS):
            continue # 금액이 아닌 단위가 붙은 숫자라면 고려하지 않음.
        elif len(valid_price_matches)>0:
            if valid_price_matches[-1].start()==match.start():
                valid_price_matches.pop()
            elif valid_price_matches[-1].end()==match.end():
                continue
            elif valid_price_matches[-1].start()<=match.start() and valid_price_matches[-1].end()>=match.end():
                continue
            elif match.start()==valid_price_matches[-1].end() and valid_price_matches[-1].group()[-1] in "원냥":
                print(valid_price_matches[-1])
                print(match)
                raise ValueError(f"regex로 한번에 못잡고 각각 따로 잡혔음.\n{valid_price_matches}\n{match}")
            elif match.start()>=valid_price_matches[-1].start() and valid_price_matches[-1].end()>match.start():
                print(valid_price_matches[-1])
                print(match)
                raise ValueError
        valid_price_matches.append(match)
    # print(filtered_matches)

    prices = [price_to_int(match.group()) for match in valid_price_matches]
    final_prices = []
    final_matches = []
    for price, match in zip(prices, valid_price_matches):
        if (price < ref_price * bottom_ratio or price > ref_price * ceil_ratio):
            # 범위를 벗어나는 경우
            if re.search(r'[원냥₩]', match.group()):
                # price가 하한선보다 작거나, 상한선보다 크더라도 가격을 나타내는 [원, ₩] 가 붙어있으면 추가
                if CATCH_DISCOUNT and bool(re.search(DISCOUNT, text[match.end():match.end()+SEARCH_DISCOUNT_WORD_UPTO])):
                    # 아직은 edge case 너무 많음
                    final_prices.append(ref_price-price)
                    final_matches.append(match)
                else:
                    final_prices.append(price)
                    final_matches.append(match)
        else:
            # 범위 내의 금액은 그냥 append
            final_prices.append(price)
            final_matches.append(match)

    return final_prices, final_matches


def price_to_int(price: str) -> int:
    """
    3만 5천과 같이 숫자와 단위가 결합된 경우, 35000으로 원래 숫자를 반환합니다.
    """
    # 필요없는 문자 제거
    price = price.replace(",", "")
    price = price.replace(" ", "")
    price = price.replace("원", "")
    price = price.replace("냥", "")
    price = price.replace("₩", "")
    price = price.replace("처넌", "천")
    price = price.replace("마넌", "만")
    price = re.sub(r"(?<!\d)0", "", price) # 앞에 붙은 0은 모두 삭제

    # case 1: 숫자로만 이뤄진 경우
    if re.match(r"^\d+$", price):
        return int(price)

    # case 2: 한글이 조금이라도 들어간 경우
    int_price = 0
    if '억' in price:
        eok, price = price.split('억')
        if eok=='':
            int_price += 100000000
        else:
            eok = str2int_under10k(eok) * 100000000
            int_price += int(eok)
    if '만' in price:
        man, price = price.split('만')
        if man=='':
            int_price += 10000
        else:
            man = str2int_under10k(man) * 10000
            int_price += int(man)
    # 10,000이하의 한글이 들어간 숫자
    int_price += int(str2int_under10k(price))
    return int_price


def str2int_under10k(num: str) -> float:
    """0 ~ 9999까지의 숫자표현을 int로 변환
    한글이 들어갈수도 있고 숫자만 들어갈수도 있음
    """
    # case 1: empty string인 경우
    if len(num)==0:
        return 0
    
    # case 2: 전부 숫자 또는 period(.)인 경우
    if re.match(r"^[\d\.]+$", num):
        return float(num)

    # case 3: 한글이 섞인 경우
    n2i={c:i for i,c in enumerate("일이삼사오육칠팔구",1)}
    unit2num={"천":1000,"백":100,"십":10}
    total = 0
    for unit in "천백십":
        if unit in num:
            n_place, num = num.split(unit)
            if n_place=='':
                total += unit2num[unit]
            elif re.match(r'^[\d\.]+$',n_place):
                total += float(n_place)*unit2num[unit]
            else:
                total += n2i[n_place]*unit2num[unit]
            if num.isdigit():
                total += int(num)
                return total
            
    if num=="":
        return total
    elif num in n2i.keys():
        total += n2i[num]
    else:
        total += float(num)
    return total


def any_string_in(strings: List[str], text: str) -> bool:
    return any([text.find(string) != -1 for string in strings])


def num2won(num:int)->str:
    """int 형으로 parsing된 숫자를 한글 문자열로 만들어줍니다."""
    units = [''] + list('만억')
    tens = [''] + list('십백천')
    result = []
    i = 0
    while num > 0:
        num, n = divmod(num, 10000)
        if n == 0:
            result.append(units[i])
        else:
            res = []
            for m in range(4):
                n, a = divmod(n, 10)
                if a == 0:
                    continue
                if m > 0:
                    res.append(tens[m])
                if a > 1 or m == 0:
                    res.append(str(a))
            result.append(''.join(reversed(res)) + units[i])
        i += 1
    return ''.join(reversed(result))+"원"

def won2num(won: str):
    return

if __name__ == "__main__":
    import random
    import json

    with open("/opt/ml/level3_nlp_finalproject-nlp-03/data/generated_train.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for i in range(3):
        d = random.choice(data)
        listed_price = d["price"]
        print(f"\nsample: {i + 1}")
        for ev in d["events"]:
            print(f"utterance: {ev['role']}: {ev['message']}")
            print(
                f"parsed price: {parse_wanted_price(ev['role'], ev['message'], listed_price, listed_price * 0.8)}"
            )
