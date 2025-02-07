import requests
import time

import schedule

from model import matching


URL_PRODUCT = 'products'
URL_GET = 'match'
URL_POST = 'match/make_match/'
API_VERSION = 'v1'
PORT = '8000'
HOST = 'localhost'


def get_address(resource: str) -> str:
    return f'http://{HOST}:{PORT}/api/{API_VERSION}/{resource}'


def get_data(url) -> dict[str, list[dict]]:
    """Функция для получения dealer_priсe и product"""
    product_data = requests.get(get_address(url))
    if product_data.status_code != 200:
        raise Exception(f' Starus: {product_data.status_code}')
    return product_data.json()


def post_data(url: str, data: list[dict]) -> None:
    """Функция для отправки итогового результата"""
    requests.post(get_address(url), json={'result': data})


def main():
    data = get_data(URL_GET)
    dealer_prices = data['dealer_priсe']
    """ XXX: Получает
    [...,
    {'id': 717, 'product_key': 'https://kub02.ru/catalog/prosept/ognebiozashchita_dlya_drevesiny_prosept_1_gruppa_s_indikatorom_gotovyy_sostav_10_kg/',
    'price': '1763.00', 'product_url': 'https://kub02.ru/catalog/prosept/ognebiozashchita_dlya_drevesiny_prosept_1_gruppa_s_indikatorom_gotovyy_sostav_10_kg/',
    'product_name': 'Огнебиозащита для древесины PROSEPT, 1 группа с индикатором, готовый состав, 10 кг', 'date': '2023-07-11', 'dealer_id': 6},
    ...]
    """

    products = data['product']
    """ XXX: Получает
    [...,
   {'id': 52, 'article': '113-075', 'ean_13': 4680008146816,
    'name': 'Средство усиленного действия для удаления ржавчины и минеральных отложений Bath Acid +  концентрат 1:200-1:500 / 0,75 л ',
    'cost': '75.00', 'min_recommended_price': '0.00', 'recommended_price': '176.00', 'category_id': 52, 'ozon_name': 'Усиленное средство для удаления ржавчины и минеральных отложений PROSEPT Bath Acid Plus, 0.75 л.',
    'name_1c': 'Усиленное средство для удаления ржавчины и минеральных отложений PROSEPT Bath Acid Plus, 0.75 л.', 'wb_name': 'Усиленное средство для удаления ржавчины и минеральных отложений PROSEPT Bath Acid Plus, 0.75 л.',
    'ozon_article': '413264558.0', 'wb_article': '149811041.0', 'ym_article': '113-075', 'wb_article_td': ''},
    ...]
    """

    result = matching(dealer_prices, products)
    post_data(URL_POST, result)
    """ Отправляет
    [...,
    {'product_key': 23, 'product_id':46}, 
    {'product_key': 23, 'product_id':49},
    ...]
    """


if __name__ == '__main__':
    # main()
    schedule.every().day.at('23:04').do(main)

    while True:
        schedule.run_pending()
        time.sleep(1)
