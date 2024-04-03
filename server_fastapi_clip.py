from fastapi import FastAPI, UploadFile
import uvicorn
from fastapi.middleware.cors import CORSMiddleware  # 추가된 부분 cors 문제 해결을 위한
import clip_model

# 모델버전
# 1. python : 3.11.5
# 2. uvicorn : 0.20.0
# 3. fastapi : 0.103.0

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

# 이미지 업로드 엔드포인트
@app.post("/")
async def upload_image(file: UploadFile, label:str):
    # 이미지 저장
    image_bytes = await file.read()
    list_labels = label.split(",")
    

    label, prob = await clip_model.predict_text_from_image(image_bytes, list_labels)    
    result = f"{label}"
    
    # JSON 데이터 출력
    return result

# Run the server
if __name__ == "__main__":
    uvicorn.run("server_test:app",
                reload = True,
                host= "127.0.0.1",
                port=8000,
                log_level="info"
                )