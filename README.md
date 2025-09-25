## 은재 ver 최초 기능 구현 코드 실행시 참고 (feature1 branch에 등록)
준비 : FlaskAPI 설치 / FFmpeg 깔고 시스템 PATH에 등록
(encode.py 코드에 ffmpeg_path로 실행경로 잘 넣어주기! 사람마다 다를 것)
https://velog.io/@tjdwjdgus99/ffmpeg-%EC%82%AC%EC%9A%A9%EB%B2%95 참고

[실행순서] 
1. server의 encode.py 실행시켜서 영상 3개 인코딩하고 segment랑 mpd 파일 생성  
2. 터미널에서 실행 (server)
cd server
python -m uvicorn main:app --reload
3. index.html 파일 Live Server로 실행




# 실시간 적응형 비디오 스트리밍 시스템  
> 네트워크 상태와 영상 특성 기반의 Adaptive Bitrate Streaming (ABR)

## 📌 프로젝트 개요
본 프로젝트는 다양한 네트워크 환경(Wi-Fi, 모바일 데이터 등)에서 발생하는 스트리밍 지연 및 품질 저하 문제를 해결하기 위해,  
클라이언트 측 네트워크 상태 및 영상 복잡도를 분석하여 서버에서 최적의 스트리밍 품질을 제공하는 실시간 적응형 스트리밍 시스템입니다.

## 🎯 주요 목표
1. **실시간 네트워크 상태 및 영상 복잡도 분석**
2. **AI 기반의 스트리밍 품질 조절**
3. **브라우저 호환 실시간 스트리밍 서비스 구현**

## 🧩 시스템 구성도
- **클라이언트**: 영상 시청, 네트워크 및 영상 특성 실시간 측정 → 서버로 전송
- **서버**: AI 모델 기반 품질 예측 → FFmpeg + HLS 스트리밍

## 🛠 기술 스택
| 분류 | 기술 |
|------|------|
| UI | HTML, CSS, JavaScript |
| 백엔드 | Python, FastAPI |
| AI 모델 | TensorFlow, Keras, OpenCV |
| 스트리밍 | FFmpeg (HLS 기반) |
| 네트워크 분석 | ntopng |
| 인프라 | Docker, AWS EC2, GitHub |
| DB | PostgreSQL |

## 📅 개발 일정
- 5~6월: 기술 학습 및 스트리밍 구현 (FFmpeg, 프론트/백엔드 개발)
- 6~7월: 영상 복잡도 분석 알고리즘 개발
- 7~8월: 네트워크 측정/분석 및 데이터 수집
- 8~9월: 실시간 적응형 스트리밍 기능 개발
- 10월: 테스트 및 개선, 최종 보고서 및 발표

## 👥 팀 소개

| 이름 | 역할 |
|------|------|
| 김남희 (202155515) | 영상 복잡도 분석 및 AI 모델 개발 |
| 박은재 (202155553) | 프론트엔드 & 백엔드 서버 개발, 스트리밍 서비스 구현 |
