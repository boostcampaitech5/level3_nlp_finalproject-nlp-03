import schedule
import time

from models import User, Chat, Product 
from datetime import datetime, timedelta 
from database import SessionLocal, dialogue_DB
from pymongo import MongoClient
from secrets import MONGODB_ID, MONGODB_PASSWORD, MONGODB_CLUSTER

## todo
# chat 형식 맞춰주기
# mongoDB itme id customizing

def load_chatData():
    db = SessionLocal()
    # 현재를 기준으로 24시간 전에 생성된 아이템은 제외하고 불러오기
    time_limit = datetime.now() - timedelta(hours=24)
    chats = db.query(Chat).filter(Chat.created_at >= time_limit).all()
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

# 동기 방식으로 mongo DataBase에 접근합니다
def update_mongoDB():
    print("running..")
    dialogue_dataset = load_chatData()

    if len(dialogue_dataset) == 0:
        print("no item to add")
        return
    
    client = MongoClient(f"mongodb+srv://{MONGODB_ID}:{MONGODB_PASSWORD}@{MONGODB_CLUSTER}.vetxlux.mongodb.net/?retryWrites=true&w=majority")

    client.ggul_tiger.dialogue.insert_many(dialogue_dataset)
    print(f"dialogue dataset updated : new data {len(dialogue_dataset)}")

# 24시간에 한번씩 함수 실행
schedule.every(24).hours.do(update_mongoDB)
# schedule.every(30).seconds.do(update_mongoDB)

while True:
    schedule.run_pending()
    time.sleep(1)
