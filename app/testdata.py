# test-data generator

# 1. db생성
# $ alembic init migrations
# app/alembic.ini       -> sqlalchemy.url = sqlite:///./app.db
# app/migrations/env.py -> import models
#                      target_metadata = models.Base.metadata

# 2. table생성
# $ alembic revision --autogenerate
# $ alembic upgrade head

# 3. test data 생성
# $ python testdata.py
#-----------------------#

from models import User, Chat, Product 
from datetime import datetime 
from database import SessionLocal


if __name__ == "__main__":
    db = SessionLocal()
    
    users = [User(username='user1', password='1234', created_at=datetime.now()),
            User(username='user2', password='1234', created_at=datetime.now()),
            User(username='user3', password='1234', created_at=datetime.now()),
            User(username='user4', password='1234', created_at=datetime.now())]
    products = [Product(title='삼성스마트티비', description='화질좋음', price=99.9, created_at=datetime.now()),
            Product(title='아이폰11', description='굿', price=78.0, created_at=datetime.now()),
            Product(title='메타몽 스티커', description='이뻐요', price=13.1, created_at=datetime.now()),
            Product(title='시디즈 T80', description='편해요', price=60.5, created_at=datetime.now()),
            Product(title='박카스', description='맛있어요', price=8.3, created_at=datetime.now())]
    chats = []

    for user in users:
        db.add(user)
    for product in products:
        db.add(product)
    for chat in chats:
        db.add(chat)
        
    db.commit()
    print(f"""--TOTAL DATA--
    users: {db.query(User).count()}
    products:{db.query(Product).count()}
    chats: {db.query(Chat).count()}""")
    db.close()
    
    print("Done")
