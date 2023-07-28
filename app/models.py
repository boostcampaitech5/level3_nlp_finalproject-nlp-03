from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float  
from sqlalchemy.orm import relationship

from database import Base 
from datetime import datetime, timezone, timedelta

# db init
# $ alembic init migrations
# models.py 변경 후 
# $ alembic revision --autogenerate
# $ alembic upgrade head

class Product(Base):
    __tablename__ ="product"
    # primary key = 고유값, 중복불가능
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    image = Column(String, default='no_image.jpeg')
    price = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=True)

class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False, unique=True)
    
    point = Column(Float, default=0.0)

    created_at = Column(DateTime, nullable=True)

class Chat(Base):
    __tablename__ = "chat"

    id = Column(Integer, primary_key=True)
    
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=True)

    user_id = Column(Integer, ForeignKey("user.id"))
    user = relationship("User", backref="chats")

    product_id = Column(Integer, ForeignKey("product.id"))
    product = relationship("Product", backref="chats")

    score = Column(String, default='없음') 

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True)
    feedback = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=True)