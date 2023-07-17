from rocketry import Rocketry
from rocketry.conds import daily, every
import time
from datetime import datetime, timedelta 
from pymongo import MongoClient

from app.models import User, Chat, Product 
from app.database import SessionLocal, dialogue_DB
from app.dbaccounts import MONGODB_ID, MONGODB_PASSWORD, MONGODB_CLUSTER
from typing import Optional
app = Rocketry()

def load_chatData(last_created:Optional[datetime]=None):
    db = SessionLocal()
    if last_created: # dialogue DB에 데이터가 존재하는 경우
        chats = db.query(Chat).filter(Chat.created_at >= last_created, Chat.created_at < datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).all() # 대화 데이터 중 가장 created_at의 날짜를 기준으로 그 이후에 생성된 데이터만 불러오기
    else: # dialogue DB에 데이터가 존재하지 않는 경우
        chats = db.query(Chat).all()
    dialogue_dataset = []
    for chat in chats:
        messages = chat.content.strip().split("\n")
        events = []
        for message in messages:
            dataset = {"role" : message[:3], "message" : message[4:]}
            events.append(dataset)
        dialogue_data = {"created_at" : chat.created_at,
                         "user_id" : chat.user_id,
                         "product_id" : chat.product_id,
                         "title" : chat.product.title,
                         "description" : chat.product.description,
                         "price" : float(chat.product.price),
                         "event" : events}
        dialogue_dataset.append(dialogue_data)

    # db 삭제
    db.commit()
    db.close()

    return dialogue_dataset


@app.task(daily.at("00:00"))
def update_dialogue():
    print("running..")
    client = MongoClient(f"mongodb+srv://{MONGODB_ID}:{MONGODB_PASSWORD}@{MONGODB_CLUSTER}.vetxlux.mongodb.net/?retryWrites=true&w=majority")
    try: # dialogue DB에 데이터가 존재하는 경우 last_created를 찾음
        last_created = client.ggul_tiger.dialogue.find().sort("created_at",-1).next()["created_at"].replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        dialogue_dataset = load_chatData(last_created)
    except:
        print("no data in dialogue DB.. ")
        dialogue_dataset = load_chatData()
    if len(dialogue_dataset) == 0:
        print("no item to add")
        return

    client.ggul_tiger.dialogue.insert_many(dialogue_dataset)
    print(f"dialogue dataset updated : new data {len(dialogue_dataset)}")

if __name__ == "__main__":
    app.run()