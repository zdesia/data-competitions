import timeit
import logging

import requests
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse

from lenta.backend import settings
from lenta.ml import model


def setup_logging():
    app_logger = logging.getLogger(__name__)
    app_logger.setLevel(logging.INFO)
    app_handler = logging.StreamHandler()
    app_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s : %(message)s")
    app_handler.setFormatter(app_formatter)
    app_logger.addHandler(app_handler)


setup_logging()
_logger = logging.getLogger(__name__)

app = FastAPI()


def get_address(resource: str) -> str:
    return f"http://{settings.API_HOST}:{settings.API_PORT}/api/{settings.API_VERSION}/{resource}/"


def get_data(data_url, params=None):
    return requests.get(data_url, params=params)


def handler(categories, shopes, sales, holiday):
    """Запуск систепы предсказания и отправка результата"""
    try:
        # Подключение системы предсказания
        result = model.forecast(categories, shopes, sales, holiday)
        _logger.info("Good forecast!")
    except Exception as err:
        _logger.error("Error forecast: " + str(err))
        raise err
    forecast_url: str = get_address(settings.URL_FORECAST)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Token {settings.TOKEN}",
    "Accept": "application/json",
    "User-Agent": "App-ML"}

    sublists = [result[i:i + settings.LIMIT_POST] for i in range(0, len(result), settings.LIMIT_POST)]
    for page in sublists:
        response = requests.post(forecast_url, headers=headers, json={"data": page})
    return _logger.info(f"Server response: {response}. Data received.")


def make_request(add: str) -> dict:
    """Получение данных с эндпоинтов"""
    data_url: str = get_address(add)
    resp = get_data(data_url)
    _logger.info(f"запрос на {data_url}")
    if resp.status_code != 200:
        _logger.error(f"Error receiving {add}")
        return []
    limit: int = resp.json()["count"]

    # В случае если данных очень много
    if limit > settings.LIMIT_REQUEST:
        resp = []
        for page in range(limit // settings.LIMIT_REQUEST):
            params = {"limit": settings.LIMIT_REQUEST, "page": page + 1}
            resp += get_data(data_url, params).json()["results"]
        return {add: resp}

    # Или если пролезаем в лимиты
    params: dict = {"limit": limit}
    resp = get_data(data_url, params=params)
    return {add: resp.json()["results"]}


# @app.get("/ml/")
# def read_root():
#     html_content = """
#             <h1>Sales forecast system.</h1>
#             <p><a href="http://127.0.0.1:9000/ml/start/">
#               Запустить систему предсказания
#             </a></p>
#             <p><a href="http://127.0.0.1:9000/docs/">
#               Документация
#             </a></p>
#             """
#     return HTMLResponse(content=html_content)


@app.get("/")
async def start(background_tasks: BackgroundTasks) -> dict:
    """Эндпоинт для запуска системы предсказания"""
    start_time = timeit.default_timer()
    try:
        categories, shopes, sales = [make_request(add) for add in settings.URL_DATA_FOR_ANALYSIS]
        holiday = {"holiday": requests.get(get_address(settings.URL_HOLIDAY)).json()}
        background_tasks.add_task(handler, categories, shopes, sales, holiday)
        _logger.info("Run handler!")
        elapsed_time = timeit.default_timer() - start_time

        # Визуальная проверка что все хорошо и данные корректные
        return {
            "status": "200",
            "tieme": elapsed_time,
            "categories": categories["categories"][0],
            "shopes": shopes["shopes"][0],
            "sales": sales["sales"][0],
            "holiday": holiday["holiday"][0]}
    except:
        return {"status": "404"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
