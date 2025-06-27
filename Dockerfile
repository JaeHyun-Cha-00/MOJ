FROM python:3.10-slim

WORKDIR /app

COPY src /app/src
COPY requirements.txt /app/

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gfortran \
    && pip install --upgrade pip \
    && pip install numpy \
    && pip install -r /app/requirements.txt

CMD ["python", "/app/src/buzzBench/chat_completion_Buzzbench.py"]
