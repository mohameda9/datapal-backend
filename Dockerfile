FROM python:3.9-slim

WORKDIR /usr/src/app
COPY ./requirements.txt .

RUN apt-get update && apt-get install -y git && \
    pip install --no-cache-dir -r requirements.txt
