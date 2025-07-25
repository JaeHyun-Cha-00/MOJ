FROM python:3.10-slim

WORKDIR /app

# 코드 복사 (src/ 안에 평가 스크립트 포함)
COPY src /app/src
COPY requirements.txt /app/

# CSV 파일 bundle
COPY data/buzzbench_converted_nautilus.csv /app/data/buzzbench_converted_nautilus.csv

# 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gfortran \
    && pip install --upgrade pip \
    && pip install numpy \
    && pip install -r /app/requirements.txt

# 실행 스크립트 설정
ENTRYPOINT ["python", "/app/src/nautilus_audience.py"]
