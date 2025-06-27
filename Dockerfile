# Python 3.10 기반 슬림 이미지 사용
FROM python:3.10-slim

# 작업 디렉토리 생성
WORKDIR /app

# 코드 및 requirements 복사
COPY src /app/src
COPY requirements.txt /app/

# 빌드 도구 및 numpy 먼저 설치 → 이후 requirements 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gfortran \
    && pip install --upgrade pip \
    && pip install numpy \
    && pip install -r /app/requirements.txt

# 기본 실행 명령어 (필요 시 수정 가능)
CMD ["python", "/app/src/buzzBench/chat_completion_Buzzbench.py"]
