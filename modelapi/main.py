from fastapi import FastAPI, Request, Depends, HTTPException
import os
from pathlib import Path
import sys

sys.path.append(Path(__file__).parent)
from load_model import load_model, convert_to_model_input
from datetime import datetime
from transformers import GenerationConfig
<<<<<<< HEAD
from logger import log
=======

>>>>>>> fa25e5e1615684a3431fc396433dd9efb766605d
path = Path(__file__).parent
app = FastAPI()
logger = log()

# 전역 변수로 모델 선언
model = None


# 모델 로드 함수 정의
def load_my_model():
    global model
    global gen_config
    model = load_model()
    # generation config를 일단은 하드코딩 해놨습니다.
    # 매 게임마다 config가 달라져도 좋을 것 같아요.
    gen_config = GenerationConfig(
        max_new_tokens=128,
        use_cahce=True,
        early_stopping=True,
        do_sample=True,
        top_k=50,
        top_p=0.85,
        num_beams=3,
        temperature=0.9,
    )
    logger.info("INFO: Model Loading Complete")


# FastAPI 앱 시작 시 모델 로드
@app.on_event("startup")
async def startup_event():
    load_my_model()
    pass


@app.get("/")
async def hello():
    return {"text": "hello"}


@app.get("/model")
async def h():
    return {"text": "hello"}


# pretrained-model 작동
@app.post("/model")
async def get_model_output(request: Request):
    data = await request.json()
    output = "hello"
    try:
        output = model.generate(convert_to_model_input(data), gen_config)
        return {"text": output}
    except Exception as e:
        print("ModelAPI:", e)
        return HTTPException(status_code=404, detail=f"{e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("modelapi.main:app", host="0.0.0.0", port=30007, reload=True)
