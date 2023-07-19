from rocketry import Rocketry
from rocketry.conds import daily, every
import time
from datetime import datetime, timedelta 
from pymongo import MongoClient

from models import User, Chat, Product 
from database import SessionLocal, dialogue_DB
from key.key import MONGODB
from logger import log 
from typing import Optional
app = Rocketry()
logger = log()

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
    print("INFO: Backup to MONGODB...")
    logger.info("Backup to MONGODB...")
    db = MONGODB()
    client = MongoClient(f"mongodb+srv://{db.MONGODB_ID}:{db.MONGODB_PASSWORD}@{db.MONGODB_CLUSTER}.vetxlux.mongodb.net/?retryWrites=true&w=majority")
    try: # dialogue DB에 데이터가 존재하는 경우 last_created를 찾음
        last_created = client.ggul_tiger.dialogue.find().sort("created_at",-1).next()["created_at"].replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        dialogue_dataset = load_chatData(last_created)
    except:
        print("no data in dialogue DB.. ")
        logger.warning("WARNING: no data in dialogue DB.. ")
        dialogue_dataset = load_chatData()
    if len(dialogue_dataset) == 0:
        print("no item to add")
        logger.warning("no item to add")
        return

    client.ggul_tiger.dialogue.insert_many(dialogue_dataset)
    print(f"Dialogue dataset updated : new data {len(dialogue_dataset)}")
    logger.debug(f"Dialogue dataset updated : new data {len(dialogue_dataset)}")

@app.task('every 1 hour')
def db_to_cloud():
    print("INFO: db to cloud: start...")
    logger.info("INFO: db to cloud: start...")
    import os
    from pathlib import Path
    path = Path(__file__).parent
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=os.path.join(path, "key/protean-iridium-391607-834302c86522.json")
    from google.cloud import storage

    storage_client = storage.Client()
    buckets = list(storage_client.list_buckets())

    bucket_name = 'sqlite-1'    # 서비스 계정 생성한 bucket 이름 입력
    source_file_name = os.path.join(path, "app.db")    # GCP에 업로드할 파일 절대경로
    destination_blob_name = 'app.db'    # 업로드할 파일을 GCP에 저장할 때의 이름


    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f'INFO: upload "{source_file_name}" to "{destination_blob_name}"')
    logger.info(f'INFO: upload "{source_file_name}" to "{destination_blob_name}"')

if __name__ == "__main__":
    app.run()