FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models/cache

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

RUN mkdir -p models/adapters logs

EXPOSE 8000

CMD ["python3", "src/api.py"]
