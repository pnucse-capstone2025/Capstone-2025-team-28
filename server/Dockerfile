# [ADDED] FastAPI 서버용 Dockerfile (경량 베이스)
FROM python:3.11-slim

# [ADDED] 시스템 의존성(필요 시 최소한만. ffmpeg 등은 나중에 추가 가능)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app/server

# [ADDED] 파이썬 패키지 설치
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# [ADDED] 애플리케이션 소스 복사
COPY . .

# [ADDED] Uvicorn으로 FastAPI 실행 (main.py의 app 객체 기준)
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
