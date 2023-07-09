from fastapi import FastAPI, Request, Form
from fastapi.param_functions import Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn  
from collections import OrderedDict
import pickle, os 
from pathlib import Path
from models import User, Product, Dialogue
from load_model import load_gpt
path = Path(__file__)

# -----------------------
# project 구조
# root
# |-- app - model/    # gpt model
#         - static/   # image, css, sample
#         - outputs/  # new_dialogue
#         - templates # html
#         - main.py   # 
#         - models.py # pydantic model
#         - load_model.py # load gpt model
# -----------------------
# todo
    # db
    # css,js ?
# -----------------------
# main_view(GET) : 중고거래 아이템 리스트
# login (GET, POST) : 로그인
# signup (GET, POST) : 회원가입
# chatting(GET, POST) : 채팅
# ranking_view(GET) : 랭킹
# -----------------------

app = FastAPI(static_url_path="app/static")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates/")

# 나중에 DB로 이동
with open('app/static/sample/users.pickle', 'rb') as f:
    users = pickle.load(f)
with open('app/static/sample/products.pickle', 'rb') as f:
    products = pickle.load(f)
with open('app/static/sample/dialogues.pickle', 'rb') as f:
    dialogues = pickle.load(f)
###

## main page
@app.get("/", description='main page', response_class=HTMLResponse)
async def main_view(request:Request):
    return templates.TemplateResponse("index.html", {"request": request, "products": products})

## signup page
@app.get("/signup", response_class=HTMLResponse)
async def signup_form(request:Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup")
async def signup(request:Request):
    form_data = await request.form()
    username = form_data["username"]
    password = form_data["password"]

    if not len(password)==4 and password.isdigit():
        return RedirectResponse(url="/signup", status_code=303,)
        
    user = User(username=username, password=password)
    users.append(user)
    return RedirectResponse(url="/login", status_code=303)

## login page
@app.get('/login', response_class=HTMLResponse)
async def login_form(request:Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post('/login')
async def login(request:Request):
    form_data = await request.form()
    username = form_data["username"]
    password = form_data["password"]

    for user in users:
        if user.username==username:
            if user.password==password:
                return RedirectResponse(url="/", status_code=303)
            
            else:
                return RedirectResponse(url="/login", status_code=303)
    return RedirectResponse(url="/login", status_code=303)

## chatting page
@app.get('/chatting/{product_id}', response_class=HTMLResponse)
async def get_chatting(request:Request, product_id:str):
    product = next((product for product in products if product.id==product_id), None)
    # output = nego_bot(
    #     title=product.title,
    #     description=product.description,
    #     price=product.price,
    #     input_text='안녕하세요.'
    # )
    dialogue = next((dialogue for dialogue in dialogues if dialogue.product.id==product.id), None).copy()
    return templates.TemplateResponse("chatting.html", {"request": request, "product": product, "chats":dialogue.chats})

@app.post("/chatting/{product_id}", response_class=HTMLResponse)
async def chatting(request:Request, product_id:str):
    form_data = await request.form()
    input_text = form_data['text']
    product = next((product for product in products if product.id==product_id), None)
    if input_text !="끝":
        dialogue = next((dialogue for dialogue in dialogues if dialogue.product.id==product.id),None).copy()
        dialogue.chats.extend([{"agent":0, "text":input_text}, {"agent":1, "text": "Okay Good"}])
        return templates.TemplateResponse("chatting.html", {"request": request, "product":product, "chats":dialogue.chats})
    else:
        return RedirectResponse(url="/", status_code=303)

## ranking page
@app.get("/ranking", response_class=HTMLResponse)
async def ranking_view(request:Request):
    user_view = [user for user in users if user.money>0] # 필터링
    user_view = sorted(user_view, key=lambda user: user.money, reverse=True) # 정렬
    return templates.TemplateResponse("ranking.html", {"request": request, "users": user_view})


if __name__=='__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True )