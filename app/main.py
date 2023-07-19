import requests
import os
import logging
from logger import log
import re 
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
from key.key import ModelRequest



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
logger = log()

# FastAPI 앱 시작 시 모델 로드
@app.on_event("startup")
async def startup_event():
    global URL, HEADERS
    model_request = ModelRequest()
    URL = model_request.REQUEST_URL
    HEADERS = model_request.REQUEST_HEADERS

## main page
@app.get("/", description="main page", response_class=HTMLResponse)
async def main_view(request: Request, db: Session = Depends(get_db)):
    logger.info(f'{request.method} "{request.url.path}" - {request.client}')

    product_list = (
        db.query(Product).order_by(Product.created_at.desc()).all()
    )  # 최신순으로 정렬
    return templates.TemplateResponse(
        "index.html", {"request": request, "products": product_list}
    )

## about page
@app.get("/about", response_class=HTMLResponse)
async def about_view(request: Request):
    logger.info(f'{request.method} "{request.url.path}" - {request.client}')

    return templates.TemplateResponse(
        "about.html", {"request": request}
    )

## signup page
@app.get("/signup", response_class=HTMLResponse)
async def signup_form(request: Request):
    logger.info(f'{request.method} "{request.url.path}" - {request.client}')

    return templates.TemplateResponse(
        "signup.html", {"request": request, "messages": []}
    )


@app.post("/signup")
async def signup(request: Request, db: Session = Depends(get_db)):
    logger.info(f'{request.method} "{request.url.path}" - {request.client}')

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
    logger.info(f'{request.method} "{request.url.path}" - {request.client}')

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
async def chatting(request: Request, product_id: int, name: str = Query(None), price=Query(None), db: Session = Depends(get_db)):
    global URL, HEADERS
    logger.info(f'{request.method} "{request.url.path}" - {request.client}')
    messages = []
    form_data = await request.form()
    if "text" in form_data.keys():
        input_text = form_data["text"]

    try:
        product = db.query(Product).filter(Product.id == product_id).first()
        current_user = db.query(User).filter(User.username==name).first()
        chat = db.query(Chat).filter(and_(Chat.user == current_user, Chat.product_id==product_id)).order_by(Chat.created_at.desc()).first()
        if price is not None and not price.isdigit():
            messages.append("정수를 입력해주세요.")
        elif price is not None and len(chat.content.strip().split("\n")) < 3:
            messages.append("대화가 더 필요합니다.")
        elif price is not None and price.isdigit() and int(price) > product.price:
            messages.append(f"{price}원에 구매할 수 없습니다.")
        elif price is not None:
            chat.content += f"구매자:##{price}##\n"
            response = requests.post(url=URL, headers=HEADERS, json=convert_to_json(chat))
            reply = response.json()['text']
            if str(response.status_code).startswith('4'):
                raise Exception("404")
            reply = re.sub(r"[^\w]", "", reply)
            # chat.content = re.sub(r"##(\d+)##", r"\1원에 구매하겠습니다.", chat.content)
            chat.content += f"판매자:{reply}\n"
            if '수락' in reply:
                point = (1.0 - float(price) / product.price) * 100
                current_user.point = round(current_user.point + point, 2)
                messages.append(f"거래성공!\n원가:{product.price}\n구매가:{price}\n네고율:{point:.2f}%")
            else:
                messages.append(f"{reply}")
            messages.append("sample")
            db.commit()
        elif len(chat.content.strip().split("\n")) >= 16:
            messages.append("최종 가격을 제안하세요.")
        elif input_text.strip() == "":
            pass
        else:
            chat.content += f"구매자:{input_text}\n"
            response = requests.post(url=URL, headers=HEADERS, json=convert_to_json(chat))
            if str(response.status_code).startswith('4'):
                raise Exception("404")
            chat.content += f"판매자:{response.json()['text']}\n"
            # chat.content += f"판매자:test\n"
            db.commit()
    except Exception as e:
         print("APP:", e)
         raise HTTPException(status_code = 404, detail= "Out of Memory")

    chats = chat.content.strip().split("\n")
    return templates.TemplateResponse(
            "chatting.html", {"request": request, "product": product, "chats": chats, "username":current_user.username, "messages":messages}
        )

# Chat -> json
def convert_to_json(chat:Chat):
    messages = chat.content.strip().split("\n")
    events = []
    for message in messages:
        event = {"role":message[:3], "message":message[4:]}
        events.append(event)
    output = {
            "title" : chat.product.title,
            "description" : chat.product.description,
            "price" : int(chat.product.price),
            "events" :  events
            }
    return output


## ranking page
@app.get("/ranking", response_class=HTMLResponse)
async def ranking_view(request: Request, all=Query(None), db: Session = Depends(get_db)):
    logger.info(f'{request.method} "{request.url.path}" - {request.client}')

    if all is None :
        user_view = db.query(User).filter(User.point>0).order_by(User.point.desc()).all()
    else:
        user_view = db.query(User).order_by(User.point.desc()).all()
    return templates.TemplateResponse(
        "ranking.html", {"request": request, "users": user_view}
    )

## feedback
@app.get("/feedback" ,response_class=HTMLResponse)
async def feedback_form(request:Request):
    logger.info(f'{request.method} "{request.url.path}" - {request.client}')
    return templates.TemplateResponse(
    "feedback.html", {"request": request}
)

@app.post("/feedback" )
async def feedback_form(request:Request, db: Session = Depends(get_db)):
    logger.info(f'{request.method} "{request.url.path}" - {request.client}')

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
    server = Server(config=uvicorn.Config("main:app", workers=1, loop = "asyncio", host="0.0.0.0", port=8000))
    api = asyncio.create_task(server.serve())
    sched = asyncio.create_task(app_rocketry.serve())

    await asyncio.wait([sched, api])

 
if __name__=='__main__':
    # logger = logging.getLogger("rocketry.task")
    # logger.addHandler(logging.StreamHandler())

    asyncio.run(main())