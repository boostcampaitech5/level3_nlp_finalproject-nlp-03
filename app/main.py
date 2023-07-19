import requests
import os
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path
path = Path(__file__).parent
import uvicorn 
import asyncio

from fastapi import FastAPI, Request, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

import sys
sys.path.append(path.parent)

from scheduler import app as app_rocketry
from models import User, Product, Chat, Feedback
from database import get_db



# -----------------------
# project 구조
# root
# |-- app - model/    # gpt model
#         - static/   # image, css, sample
#         - outputs/  # new_dialogue
#         - templates # html
#         - main.py   # fastapi
#         - models.py # db 데이터 포맷 설계
# -----------------------
# todo
# async db
# user authentication(login)
# front
# -----------------------
# main_view(GET) : 중고거래 아이템 리스트
# login (GET, POST) : 로그인(x)
# signup (GET, POST) : 회원가입
# chatting(GET, POST) : 채팅
# ranking_view(GET) : 랭킹
# -----------------------

app = FastAPI(static_url_path=os.path.join(str(path), "static"))
app.mount(
    "/static", StaticFiles(directory=os.path.join(str(path), "static")), name="static"
)
templates = Jinja2Templates(directory=os.path.join(str(path), "templates"))

session = app_rocketry.session


# FastAPI 앱 시작 시 모델 로드
@app.on_event("startup")
async def startup_event():
    global URL, HEADERS
    URL = 'https://safely-expert-lobster.ngrok-free.app/model'
    HEADERS = {'ngrok-skip-browser-warning': 'true'}

## main page
@app.get("/", description="main page", response_class=HTMLResponse)
async def main_view(request: Request, db: Session = Depends(get_db)):
    product_list = (
        db.query(Product).order_by(Product.created_at.desc()).all()
    )  # 최신순으로 정렬
    return templates.TemplateResponse(
        "index.html", {"request": request, "products": product_list}
    )


## signup page
@app.get("/signup", response_class=HTMLResponse)
async def signup_form(request: Request):
    return templates.TemplateResponse(
        "signup.html", {"request": request, "messages": []}
    )


@app.post("/signup")
async def signup(request: Request, db: Session = Depends(get_db)):
    form_data = await request.form()
    username = form_data["username"]

    check = db.query(User).filter(User.username == username).all()
    if check:
        return templates.TemplateResponse(
            "signup.html", {"request": request, "messages": ["이미 존재하는 이름입니다."]}
        )
    new_user = User(username=username)
    db.add(new_user)
    db.commit()

    return RedirectResponse(url="/", status_code=303)


## login page
# @app.get("/login", response_class=HTMLResponse)
# async def login_form(request: Request):
#     return templates.TemplateResponse(
#         "login.html", {"request": request, "messages": []}
#     )


# @app.post("/login")
# async def login(request: Request, db: Session = Depends(get_db)):
#     form_data = await request.form()
#     username = form_data["username"]
#     password = form_data["password"]

#     user = db.query(User).filter(User.username == username).all()
#     if not user:
#         return templates.TemplateResponse(
#             "login.html", {"request": request, "messages": ["아이디가 없습니다."]}
#         )  # 아이디오류
#         # pass
#     if isinstance(user, list):
#         user = user[0]
#     if str(user.password) != str(password):
#         return templates.TemplateResponse(
#             "login.html", {"request": request, "messages": ["비밀번호가 틀렸습니다."]}
#         )  # 비밀번호 오류
#     return RedirectResponse(url="/", status_code=303)


## chatting page
@app.get("/chatting/{product_id}", response_class=HTMLResponse)
async def get_chatting(
    request: Request, product_id: int, name: str = Query(None), db: Session = Depends(get_db)
):
    product = db.query(Product).filter(Product.id == product_id).first()
    current_user = db.query(User).filter(User.username==name).first()
    if not current_user:
        return RedirectResponse(url="/signup", status_code=303)
    new_chat = Chat(
        content="",
        created_at=datetime.now(),
        user=current_user,
        product=product,
    )
    db.add(new_chat)
    db.commit()
    return templates.TemplateResponse(
        "chatting.html", {"request": request, "product": product, "username":current_user.username}
    )


@app.post("/chatting/{product_id}", response_class=HTMLResponse)
async def chatting(request: Request, product_id: int, name: str = Query(None), db: Session = Depends(get_db)):
    global URL, HEADERS
    form_data = await request.form()
    input_text = form_data["text"]

    try:
        product = db.query(Product).filter(Product.id == product_id).first()
        current_user = db.query(User).filter(User.username==name).first()
        chat = db.query(Chat).filter(and_(Chat.user == current_user, Chat.product_id==product_id)).order_by(Chat.created_at).first()
        print(product, current_user, chat)
        if input_text.strip() == "":
            pass
        elif input_text == "끝":
            chat.content += f"구매자:{input_text}"
            db.commit()
            if len(chat.content.strip().split("\n")) <= 2:
                db.delete(chat)  # 대화 턴이 짧으면 삭제
                db.commit()
            return RedirectResponse(url="/", status_code=303)
        else:
            chat.content += f"구매자:{input_text}\n"
            # response = requests.post(url=URL, headers=HEADERS, json=convert_to_json(chat))
            # if str(response.status_code).startswith('4'):
            #     raise Exception("404")
            # chat.content += f"판매자:{response.json()['text']}\n"
            chat.content += f"판매자:hello\n"
            db.commit()
    except Exception as e:
         print("APP:", e)
         raise HTTPException(status_code = 404, detail= "Out of Memory")

    chats = chat.content.strip().split("\n")
    return templates.TemplateResponse(
        "chatting.html", {"request": request, "product": product, "chats": chats, "username":current_user.username}
    )

# Chat -> json
def convert_to_json(chat:Chat):
    messages = chat.content.strip().split("\n")
    events = []
    for message in messages:
        event = [message[:3], message[4:]]
        events.append(event)
    output = {
            "title" : chat.product.title,
            "description" : chat.product.description,
            "price" : float(chat.product.price),
            "events" :  events
            }
    return output


## ranking page
@app.get("/ranking", response_class=HTMLResponse)
async def ranking_view(request: Request, db: Session = Depends(get_db)):
    user_view = db.query(User).filter(User.money >= 0).order_by(User.money.desc()).all()
    return templates.TemplateResponse(
        "ranking.html", {"request": request, "users": user_view}
    )

## feedback
@app.get("/feedback" ,response_class=HTMLResponse)
async def feedback_form(request:Request):
        return templates.TemplateResponse(
        "feedback.html", {"request": request}
    )

@app.post("/feedback" )
async def feedback_form(request:Request, db: Session = Depends(get_db)):
        form_data = await request.form()
        feedback = form_data['feedback'] # str
        ## feedback 저장
        db.add(Feedback(feedback=feedback))
        db.commit()
        return templates.TemplateResponse(
        "index.html", {"request": request}
    )

@app.get("/logs")
async def read_logs():
    "schduled task의 log를 불러옵니다"
    repo = session.get_repo()
    return repo.filter_by().all()

# server shutdown 시 전부 닫을 수 있도록 재정의
class Server(uvicorn.Server):
    def handle_exit(self, sig : int, format : Optional[str]) -> None:
        print("shutting down all task")
        app_rocketry.session.shut_down()
        return super().handle_exit(sig, format)

## main 함수
async def main():
    server = Server(config=uvicorn.Config("main:app", workers=1, loop = "asyncio", host="0.0.0.0", port=8080))
    api = asyncio.create_task(server.serve())
    sched = asyncio.create_task(app_rocketry.serve())

    await asyncio.wait([sched, api])

 
if __name__=='__main__':
    logger = logging.getLogger("rocketry.task")
    logger.addHandler(logging.StreamHandler())

    asyncio.run(main())