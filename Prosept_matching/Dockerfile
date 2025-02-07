FROM python:3.10.9-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip &&\
    pip install -r requirements.txt --no-cache-dir

CMD ["python", "app.py"]