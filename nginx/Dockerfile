# ============================
# nginx/Dockerfile  (전체 교체)
# ============================

# [STAGE 1] React 프론트 빌드
FROM node:18-alpine AS build
WORKDIR /app

# [ADDED] 프론트 패키지 파일만 먼저 복사 후 설치 (레이어 캐시 최적화)
#  - compose의 build.context를 프로젝트 루트(.)로 설정해야 아래 경로들이 보임
#  - Netflix 폴더명은 현재 구조를 그대로 사용
COPY Netflix/package*.json ./
RUN npm ci

# [ADDED] 소스 전체 복사 후 빌드
COPY Netflix/ ./
RUN npm run build  # => /app/dist 생성

# [STAGE 2] Nginx 정적 서빙
FROM nginx:1.27-alpine

# [ADDED] our nginx config
#  - compose에서 동일 경로를 볼륨 마운트(RO)해도 안전
COPY nginx/default.conf /etc/nginx/conf.d/default.conf

# [ADDED] 빌드 산출물을 Nginx 정적 루트로 복사
COPY --from=build /app/dist /usr/share/nginx/html

# (선택) 헬스체크나 추가 설정 필요 시 여기에 작성
# EXPOSE 80
# EXPOSE 443

