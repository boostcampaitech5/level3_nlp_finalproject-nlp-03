#!pip install googletrans==3.1.0a0
#!pip install tqdm

from googletrans import Translator
import json
from tqdm.auto import tqdm
from collections import OrderedDict

def trans_func(translator, index, data):
    ## 필요한 정보만 추출
    # category = data[index]["scenario"]["category"]
    # buyer_target_price = data[index]["scenario"]["kbs"][0]["personal"]["Target"]
    # seller_target_price = data[index]["scenario"]["kbs"][1]["personal"]["Target"]
    description = translator.translate(data[index]["scenario"]["kbs"][0]["item"]["Description"], src='en', dest='ko')
    title = translator.translate(data[index]["scenario"]["kbs"][0]["item"]["Title"], src='en', dest='ko')
    # agents = data[index]["agents"]
    # outcome = data[index]["outcome"]
    events = data[index]["events"]
    for i, e in enumerate(events):
        if e['action'] == 'message':
            events[i]['data'] = translator.translate(e['data'], src='en',dest='ko').text
        # else:
        #     events.append(f"{e['action']}: {e['data']}")

    ## 번역텍스트만 추출
    description = [result.text for result in description]
    title = [result.text for result in title] if isinstance(title, list) else title.text

    ## 저장
    data[index]["scenario"]["kbs"][0]["item"]["Description"] = description
    data[index]["scenario"]["kbs"][0]["item"]["Title"] = title
    data[index]["scenario"]["kbs"][1]["item"]["Description"] = description
    data[index]["scenario"]["kbs"][1]["item"]["Title"] = title
    data[index]["events"] = events
    return data


if __name__=='__main__':
    # translator
    translator = Translator()

    # filename
    data_path = './level3/data/dev.json'
    save_path = './level3/data/output.json'

    # load
    with open(data_path, 'r') as f:
        data = json.load(f)

    # traslating
    for i in tqdm(range(0, 100), desc='tranlating'):
        data = trans_func(translator, i, data)
    
    # save
    with open(save_path, mode='w', encoding="utf-8") as f:
        f.write(
            json.dumps(data, indent=4, ensure_ascii=False) + "\n"
        )
            




