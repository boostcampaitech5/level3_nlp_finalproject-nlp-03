from sqlalchemy import create_engine 
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# mongoDB 연결을 위한 라이브러리
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from key.key import MONGODB
from bson.objectid import ObjectId

# db접속주소, sqlite3 db파일, 프로젝트 루트디렉토리
SQLALCHEMY_DATABASE_URL = 'sqlite:///./app.db'

# 커넥션 풀 생성: db에 접속하는 객체를 일정 갯수만큼 만들어 놓고 돌려가며 사용
engine=create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={'check_same_thread':False}
)

# db에 접속하기 위해 필요한 클래스
# autocommit=False -> 데이터변경후 commit을 해야만 실제로 저장, 잘못저장했을때 rollback가능
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# db모델을 구성할 때 사용하는 클래스
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# mongoDB에 비동기식으로 접근하고자 할 때 활용합니다
# database, collection 명은 고정되어있는 상태입니다. 필요시 데이터베이스를 새로 만들고 수정하여 활용합니다.
class MongoDB:
    def __init__(self):
        self.client = None
        self.engine = None

    def connect(self):
        db = MONGODB()
        self.client = AsyncIOMotorClient(f"mongodb+srv://{db.MONGODB_ID}:{db.MONGODB_PASSWORD}@{db.MONGODB_CLUSTER}.vetxlux.mongodb.net/?retryWrites=true&w=majority")
        print("Mongo DB와 비동기적으로 연결되었습니다.")
    
    def close(self):
        self.client.close()

    async def create_item(self, user, product):
        new_dialogue = {"user_id" : user.id,
                       "product_id" : product.id,
                       "title" :product.title, 
                       "description" : product.description, 
                       "price" : float(product.price),
                       "events": []
                       }
        try:
            result = await self.client.ggul_tiger.dialogue.insert_one(new_dialogue)
            return result.insert_id
        except Exception as error:
            print("Error: ", error)
            print("\t cannot create mongodb item")
            return error
        
    async def update_item(self, dialogue_id:ObjectId, user, product, chat:str):
        new_events = await self.client.ggul_tiger.dialogue.find({"_id":dialogue_id})
        chat = chat.strip().split("\n")
        for message in chat:
            dataset = {"role" : message[:3], "message" : message[4:]}
            new_events.append(dataset)
        try : 
            result = await self.client.ggul_tiger.dialogue.update_one({"_id":dialogue_id}, {"$set":{"event" : new_events}})
            return result.insert_id
        except Exception as error:
            print("Error: ", error)
            print("\t cannot update mongodb item")
            return error

dialogue_DB = MongoDB()