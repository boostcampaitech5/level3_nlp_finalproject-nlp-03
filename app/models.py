from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Numeric  
from sqlalchemy.orm import relationship

from database import Base 
from datetime import datetime

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
    created_at = Column(DateTime, default=datetime.now())

class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False, unique=True)
    money = Column(Integer, default=100)
    created_at = Column(DateTime, default=datetime.now())

class Chat(Base):
    __tablename__ = "chat"

    id = Column(Integer, primary_key=True)
    
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now())

    user_id = Column(Integer, ForeignKey("user.id"))
    user = relationship("User", backref="chats")

    product_id = Column(Integer, ForeignKey("product.id"))
    product = relationship("Product", backref="chats")

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True)
    feedback = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now())