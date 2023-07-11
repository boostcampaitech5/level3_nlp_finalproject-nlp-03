from fastapi import FastAPI, Request, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn  
from pathlib import Path
from models import User, Product, Chat
from database import get_db
from load_model import load_gpt
from datetime import datetime 
path = Path(__file__)

# -----------------------
# project 구조
# root
# |-- app - model/    # gpt model
#         - static/   # image, css, sample
#         - outputs/  # new_dialogue
#         - templates # html
#         - main.py   # fastapi
#         - models.py # db 데이터 포맷 설계
#         - load_model.py # load gpt model
# -----------------------
# todo
    # async db
    # user authentication(login)
    # front
# -----------------------
# main_view(GET) : 중고거래 아이템 리스트
# login (GET, POST) : 로그인
# signup (GET, POST) : 회원가입
# chatting(GET, POST) : 채팅
# ranking_view(GET) : 랭킹
# -----------------------

app = FastAPI(static_url_path="static/")
app.mount("/static", StaticFiles(directory="static/"), name="static")
templates = Jinja2Templates(directory="templates/")

## main page
@app.get("/", description='main page', response_class=HTMLResponse)
async def main_view(request:Request, db:Session=Depends(get_db)):
    product_list = db.query(Product).order_by(Product.created_at.desc()).all() # 최신순으로 정렬
    return templates.TemplateResponse("index.html", {"request": request, "products": product_list})

## signup page
@app.get("/signup", response_class=HTMLResponse)
async def signup_form(request:Request):
    return templates.TemplateResponse("signup.html", {"request": request, "messages":[]})

@app.post("/signup")
async def signup(request:Request, db:Session=Depends(get_db)):
    form_data = await request.form()
    username = form_data["username"]
    password = form_data["password"]

    check = db.query(User).filter(User.username==username).all()
    if check:
        return templates.TemplateResponse("signup.html", {"request": request, "messages":['이미 존재하는 이름입니다.']})
    new_user = User(username=username,
                    password=password,
                    created_at=datetime.now())
    db.add(new_user)
    db.commit()

    return RedirectResponse(url="/login", status_code=303)

        
    
## login page
@app.get('/login', response_class=HTMLResponse)
async def login_form(request:Request):
    return templates.TemplateResponse("login.html", {"request": request, "messages":[]})

@app.post('/login')
async def login(request:Request, db:Session=Depends(get_db)):
    form_data = await request.form()
    username = form_data["username"]
    password = form_data["password"]

    user = db.query(User).filter(User.username==username).all()
    if not user:
        return templates.TemplateResponse("login.html", {"request": request, "messages":["아이디가 없습니다."]}) # 아이디오류
        # pass
    if isinstance(user, list):
        user = user[0]
    if user.password != password:
        return templates.TemplateResponse("login.html", {"request": request, "messages":['비밀번호가 틀렸습니다.']})# 비밀번호 오류
    return RedirectResponse(url="/", status_code=303)

## chatting page
@app.get('/chatting/{product_id}', response_class=HTMLResponse)
async def get_chatting(request:Request, product_id:int, db:Session=Depends(get_db)):
    product = db.query(Product).filter(Product.id==product_id).first()
    sample_user = db.query(User).filter(User.id==1).first() # 임시
    new_chat = Chat(content='',
                    created_at=datetime.now(),
                    user=sample_user, # 로그인 구현후에 request.user (현재유저)로 변경
                    product=product)
    db.add(new_chat)
    db.commit()
    return templates.TemplateResponse("chatting.html", {"request": request,"product":product})

@app.post("/chatting/{product_id}", response_class=HTMLResponse)
async def chatting(request:Request, product_id:int, db:Session=Depends(get_db)):
    form_data = await request.form()
    input_text = form_data['text']
    
    product = db.query(Product).filter(Product.id==product_id).first()
    sample_user = db.query(User).filter(User.id==1).first() # 임시
    chat = db.query(Chat).filter(and_(Chat.user==sample_user)).all()[-1]

    if input_text.strip()=="":
        pass
    elif input_text == '끝':
        chat.content += f"구매자:{input_text}"
        db.commit()
        if len(chat.content.strip().split("\n")) <= 3: db.delete(chat) # 대화 턴이 짧으면 삭제
        return RedirectResponse(url="/", status_code=303)
    else:
        output = '네 좋아요' # 모델output
        chat.content += f"구매자:{input_text}\n판매자:{output}\n"
        db.add(chat)
        db.commit()

    chats = chat.content.strip().split("\n")
    return templates.TemplateResponse("chatting.html", {"request": request, "product":product, "chats":chats})


## ranking page
@app.get("/ranking", response_class=HTMLResponse)
async def ranking_view(request:Request, db:Session=Depends(get_db)):
    user_view = db.query(User).filter(User.money >=0 ).order_by(User.money.desc()).all()
    return templates.TemplateResponse("ranking.html", {"request": request, "users": user_view})

if __name__=='__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)