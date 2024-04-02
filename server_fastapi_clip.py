import uvicorn  # pip install uvicorn
from fastapi import FastAPI  # pip install fastapi
from fastapi.middleware.cors import CORSMiddleware  # 추가된 부분 cors 문제 해결을 위한
from pydantic import BaseModel

# 모델버전
# 1. python : 3.11.5
# 2. uvicorn : 0.20.0
# 3. fastapi : 0.103.0
# 4. pydantic : 1.10.8

# 수신 json형태
# {"path" : "test.jpg", "labels" : "cat, dog, tiger, pig"}
# path + label로 받는 형태

# 예측 모듈 가져오기
import mmd_clip_test2

class get_json(BaseModel):
    path : str
    labels : str

# FastAPI application
app = FastAPI()

# cors 이슈
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"], ## 모든 헤더 허용
)

@app.post("/")
async def read_root(jsn:get_json):
    
    # labels 받기
    labels = jsn.labels.split(",")
    
    # url 받기
    path = jsn.path
    
    # 모델 예측하기
    label, prob = await clip_model.predict_text_from_image(path, labels)    
    result = f"label : {label}, prob : {str(prob)}"

    # json 리턴
    # {"label" : "cat", "prob" : "0.65454"}
    return result

# Run the server
if __name__ == "__main__":
    uvicorn.run("server_fastapi_clip:app",
                reload = True,
                host= "127.0.0.1",
                port=8000,
                log_level="info"
                )
