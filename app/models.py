from pydantic import BaseModel, Field
from uuid import uuid4
from typing import List, Union, Optional, Dict, Any, Type
from datetime import datetime 

class User(BaseModel):
    id: str = Field(default_factory=str(uuid4())[:8])
    username : str 
    password : str
    money : int = Field(default_factory=100)
    dialogue_id : List[str] = Field(default_factory=list)
    # dialogue : List[Type['Dialogue']] = Field(default_factory=list)

    # def add_dialogue(self, dialogue:Type['Dialogue']):
    #     # 검사?
    #     if False:
    #         return self
        
    #     self.dialogue.append(dialogue)
    #     return self
    
class Product(BaseModel):
    id: str = Field(default_factory=str(uuid4())[:8])
    image:Any = 'no_image.jpeg'
    title: str 
    description: str 
    price: float

class Dialogue(BaseModel):
    id: str = Field(default_factory=str(uuid4())[:8])
    user: User
    product: Product
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    chats: List[Dict]