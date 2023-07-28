import sys
sys.path.append(".")
from chat_bot.neural_chat.craigslist.price_parser import parse_prices, price_to_int
from unittest import TestCase
import unittest

def num2kor(num:int):
    units = [''] + list('만억조')
    nums = '일이삼사오육칠팔구'
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
                    res.append(nums[a-1])
            result.append(''.join(reversed(res)) + units[i])
        i += 1
    return ''.join(reversed(result))

class TestParsePrice(TestCase):
    def test_parse_prices(self):
        self.assertEqual(parse_prices("삼천만원은 너무 비싼데요? 50프로만 깎아줘요",30000000,0.5,2)[0],[30000000])
        self.assertEqual(parse_prices("마넌에 사고싶어요",10000,0.5,2)[0],[10000])
        self.assertEqual(parse_prices("그냥 오처넌에 헤주세요",5000,0.5,2)[0],[5000])
        self.assertEqual(parse_prices("삼처넌",3000,0.5,2)[0],[3000])
        self.assertEqual(parse_prices("3천만원은 너무 비싼데요?",30000000,0.5,2)[0],[30000000])
        self.assertEqual(parse_prices("삼천만원은 너무 비싼데요?",30000000,0.5,2)[0],[30000000])
        self.assertEqual(parse_prices("삼천육백만원은 너무 비싼데요?",36000000,0.5,2)[0],[36000000])
        self.assertEqual(parse_prices("3000은 어떠세요?",3000,0.5,2)[0],[3000])
        self.assertEqual(parse_prices("오처넌에 해주시면 안돼요?",5000,0.5,2)[0],[5000])
        self.assertEqual(parse_prices("10마넌",100000,0.5,2)[0],[100000])
        self.assertEqual(parse_prices("1억 3000만원에 살게여",130000000,0.5,2)[0],[130000000])
        self.assertEqual(parse_prices("₩5,000",5000,0.5,2)[0],[5000])
        
    def test_price2int(self):
        self.assertEqual(price_to_int(f"천만"),10000000)
        self.assertEqual(price_to_int(f"백만"),1000000)
        self.assertEqual(price_to_int(f"십만"),100000)
        self.assertEqual(price_to_int(f"만"),10000)
        self.assertEqual(price_to_int(f"천"),1000)
        self.assertEqual(price_to_int(f"백"),100)
        self.assertEqual(price_to_int(f"십"),10)
        for i in range(1,10):
            self.assertEqual(price_to_int(f"{i}천만"),i*10000000)
            self.assertEqual(price_to_int(f"{i}백만"),i*1000000)
            self.assertEqual(price_to_int(f"{i}십만"),i*100000)
            self.assertEqual(price_to_int(f"{i}만"),i*10000)
            self.assertEqual(price_to_int(f"{i}천"),i*1000)
            self.assertEqual(price_to_int(f"{i}백"),i*100)
            self.assertEqual(price_to_int(f"{i}십"),i*10)
        self.assertEqual(price_to_int("3천 5백"),3500)
        self.assertEqual(price_to_int("3천 5백만"),35000000)
        
        # for i in range(1,100000000):
        #     self.assertEqual(price_to_int(num2kor(i)),i)

if __name__=="__main__":
    unittest.main()