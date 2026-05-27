FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV FLASK_APP=server.py \
    PROJ_DIR=. \
    LOG_LEVEL=debug \
    PYTHONUNBUFFERED=1

LABEL author="Roman Tsvetkov" version="0.0.1"

WORKDIR /app
COPY $PROJ_DIR /app

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python3", "server.py"]