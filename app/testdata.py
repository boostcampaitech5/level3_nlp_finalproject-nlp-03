# test-data generator
from models import User, Chat, Product , Feedback
from datetime import datetime 
from database import SessionLocal
import json
import os 
from pathlib import Path


if __name__ == "__main__":
    db = SessionLocal()
    with open(os.path.join(Path(__file__).parent, "key/sample_data.json"), 'r') as f:
        data = json.load(f)

    for user in data['user']:
        db.add(User(username=user['username']))
    for product in data['product']:
        db.add(Product(title=product['title'],
                       description=product['description'],
                       price=product['price'],
                       image=product['image']))
    # for chat in data['chat']:
    #     db.add(Ch)
    for feedback in data['feedback']:
        db.add(Feedback(feedback=feedback['feedback']))
    db.commit()

    print(f"""--TOTAL SAMPLE DATA--
    users: {db.query(User).count()}
    products:{db.query(Product).count()}
    chats: {db.query(Chat).count()}
    feedback: {db.query(Feedback).count()}""")
    db.close()
    
    print("Done")
