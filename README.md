[<img width="600" alt="image" src="https://github.com/boostcampaitech5/level3_nlp_finalproject-nlp-03/assets/75467530/37831d49-2e42-46ca-bcae-6f9caeaa934b">](https://boostcamp.connect.or.kr/)

# :honey_pot:NELLM(낼름): NEgotiation Large Language Model

NELLM(낼름)은 중고거래 상황에서 가격을 협상하는 챗봇 어플리케이션입니다.  
**판매자의 역할**을 수행하여 구매자의 채팅에 적절한 대답을 합니다.

# 🌱Members

|<img src='https://avatars.githubusercontent.com/u/110003154?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/60145579?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/54995090?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/75467530?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/65614582?v=4' height=100 width=100px></img>|
| --- | --- | --- | --- | --- |
| [김민혁](https://github.com/torchtorchkimtorch) | [김의진](https://github.com/KimuGenie) | [김성우](https://github.com/tjddn0402) | [오원택](https://github.com/dnjdsxor21) | [정세연](https://github.com/jjsyeon) |

# Demo & Example
<http://nellm.site/>

<img width="935" alt="image" src="https://github.com/boostcampaitech5/level3_nlp_finalproject-nlp-03/assets/75467530/c0a6bc7c-e8c2-4bd0-a45d-78e89a39899b">

# How to run
1. WebAPI: 낼름의 프론트 & 백엔드를 구성하는 API
**WebAPI**
```bash
cd app
uvicorn main:app --port 80
```
2. ModelAPI: 구매자의 채팅을 입력 받아 적절한 대답을 출력하는 API
**ModelAPI**
```bash
cd modelapi
uvicorn main:app --port 30007
```

# Model
[NELLM(낼름)](https://huggingface.co/ggul-tiger)은 [KULLM(구름)](https://github.com/nlpai-lab/KULLM)을 바탕으로 Fine-tuning된 모델입니다.  

# Dataset
[낼름 데이터셋 v1](https://huggingface.co/ggul-tiger): ChatGPT로 자체 생성한 데이터셋  
[낼름 데이터셋 v2](https://huggingface.co/ggul-tiger): + 앱 배포 후 사용자로부터 얻은 데이터셋  


