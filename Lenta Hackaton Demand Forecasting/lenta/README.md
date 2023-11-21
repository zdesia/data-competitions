## Инструкция по запуску на локальной машине.
Клонировать репозиторий

```git clone git@github.com:aminaadzhieva/lenta-hackathon-demand-forecasting.git```

Установить виртуальное окружение

```python3.11 -m venv venv```

Запустить виртуальное окружение

```source venv/bin/activate```

Установить все зависимости

```pip install -r requirements.txt```

Запустить сервер

```uvicorn lenta.backend.app:app```

[Запустить систему предсказания](http://127.0.0.1:8000/api/v1/forecast/custom_response_post/)


[Проверить данные в админке](http://31.129.109.228/admin/)