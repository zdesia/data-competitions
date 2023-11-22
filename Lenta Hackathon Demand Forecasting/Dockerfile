FROM python:3.10.6

WORKDIR /app

COPY lenta/requirements.txt .

RUN apt-get update && apt-get install -y nano && pip3 install --upgrade pip && pip3 install -r /app/requirements.txt --no-cache-dir

COPY ./lenta /app/lenta

CMD ["uvicorn", "lenta.backend.app:app", "--host", "0.0.0.0", "--port", "9000"]
