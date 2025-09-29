#FastAPI 서버
#pip install fastapi uvicorn 로 설치 먼저 하세요!!
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# 클라이언트에서 접근 가능하도록 CORS 허용 (다른 포트에서 접근 가능하게)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    #allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



app.mount(
    "/stream",
    StaticFiles(directory="static"),
    name="stream",
)


# 절대 경로 기준으로 static 마운트
#위에 코드 안되면 이 경로 사용
'''
base_dir = os.path.dirname(os.path.abspath(__file__))
static_path = os.path.join(base_dir, "static")

app.mount("/", StaticFiles(directory=static_path), name="static")
'''