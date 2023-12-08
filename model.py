import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pymorphy3
import joblib
import tfidf_matcher as tm
from catboost import CatBoostClassifier, Pool
import sys
import json

nltk.download('stopwords')
nltk.download('punkt')

import warnings
warnings.filterwarnings('ignore')

pd.DataFrame.iteritems = pd.DataFrame.items # совместимостоь text_features в Pool и pandas

# число рекомендаций для грубого поиска
K_RUDE = 20
# количество выводимых рекомендаций
K_PRECISE = 5

MODEL_CB = joblib.load('model_clf_cb.pkl')

def matching(dealer_json: list, products_json: list):
    
    if isinstance(products_json, list):
        data_products = pd.DataFrame(products_json)
    if isinstance(dealer_json, list):
        data_dealers = pd.DataFrame(dealer_json)

    # подготовка датасетов
    # удалить продукты дубликаты, спарсенные в разные дни. Перезаписать и ндексы
    data_dealers = data_dealers[
        ~data_dealers.drop(['id', 'date', 'product_url', 'price'], axis=1)
        .duplicated()
    ].reset_index(drop=True)
    
    # пропуски в столбце name заполнить значениями в столбце name_1c (если есть)
    data_products['name'] = data_products['name'].fillna(data_products['name_1c'])

    # удалить пропуски в столбце с именем
    data_products = data_products.dropna(subset=['name']).reset_index(drop=True)



    # Функция очистки,  токенизации, лемматизации и удаление стоп-слов.
    def preprocess_text(text):
        # добавить пропущенные пропуски
        text = re.sub(r'(?<=[a-zA-Z])(?=[а-яА-ЯёЁ])|(?<=[а-яА-ЯёЁ])(?=[a-zA-Z])', ' ', text)
        text = re.sub(r'(\S)\*(\S)', r'\1 * \2', text)
        text = re.sub(r'(\d+)([а-яА-ЯёЁa-zA-Z]+)', r'\1 \2', text)

        text = re.sub(r'\b\d+-\d+\b', '', text) # Удаление артикула типа "152-5" в конце строки  r'\s*\d+-\d+$'

        text = text.lower()

        # в наименования оказались еще места, которые не были обработаны функцией и уберем эти проблемы вручную
        text = (text
               .replace('с редство', 'средство')
               .replace('бетонконтактготовый', 'бетонконтакт готовый')
               .replace('"к', '" к')
               .replace('яблокаконцентрированное', 'яблока концентрированное')
               .replace('(сухой остаток 20%)', ' (сухой остаток 20%) ')
               .replace('цитрусаконцентрат', 'цитруса концентрат')
               .replace('.с', '. с')
               .replace('0концентрат', '50 концентрат')
               )

        text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9 ,.:]", ' ', text)
        text = re.sub(r"[,]", '.', text) 
        text = re.sub(r'\.\s', ' ', text) # если точка после пробела

        # уберём название производителя
        text = text.replace('prosept', '').replace('просепт', '')

        # Токенизация
        tokens = word_tokenize(text)

        # лемматизация русских слов
        morph = pymorphy3.MorphAnalyzer()
        lemmas = [morph.parse(word)[0][2] for word in tokens]

        # Удаление стоп-слов - улучшает
        stop_words = set(stopwords.words('russian') + stopwords.words('english'))
        filtered_words = [lemma for lemma in lemmas if lemma not in stop_words]

        # Возвращение предобработанного текста
        cleaned_name =  ' '.join(filtered_words)

        # стандартизация порядка написания в названии объёма и тд - ухудшает
        feature =  cleaned_name #+ ' ' + concent + ' ' + volume + ' ' + units_of_meas + ' ' + units

        return feature

    # очистка текста названий производителя
    data_products['feature'] =  data_products['name'].apply(preprocess_text)
    # очистка текста названий с сайтов дилеров
    data_dealers['dealer_feature'] =  (data_dealers['product_name'].apply(preprocess_text))


    # 1. Грубый поиск: Модель `TFIDF-matcher` предсказывает 20 рекомендаций (генерирует таблицу)
    tfidf_match_table = tm.matcher(
        original=data_dealers['dealer_feature'], 
        lookup=data_products['feature'],
        k_matches=K_RUDE,
        ngram_length=3
    )
    
    # 2. Пересобрать 20 предсказний в датасет с парами <Название дилера - Рекомендованной название> 
    records = []

    # перебор строк в DataFrame
    for i, row in tfidf_match_table.iterrows():
        original_name = row['Original Name']

        # перебор рекомендаций в столбцах таблицы tfidf-matcher
        for col_name, col_index in zip(['Lookup ' + str(i) for i in range(1, K_RUDE+1)], 
                       ['Lookup ' + str(i) + ' Index' for i in range(1, K_RUDE+1)]):
            recommendation = row[col_name]
            product_key = data_dealers.loc[i, 'product_key']
            rec_product_index = row[col_index]                          
            rec_product_id = data_products.loc[rec_product_index, 'id']
        
            records.append({'dealer_feature': original_name, 'recommendation': recommendation,
                           'product_key': product_key, 'product_id':rec_product_id})

    pair_recommendation_df = pd.DataFrame(records)
    
    # 3. Точный поиск: Сделать предсказание вероятности обученным CatBoost
    data_pool = Pool(pair_recommendation_df[['dealer_feature', 'recommendation']],
                     text_features=['dealer_feature', 'recommendation'])

    # расссчитываем вероятности
    pair_recommendation_df['probability'] = MODEL_CB.predict_proba(data_pool)[:, 1]
        
    # 4. Взять 5 рекомендаций с самой высокой вероятностью
    # cортировка по вероятности, группировка по 'product_key' и выбор 5 самых высоких вероятностей в каждой группе
    result_df = (pair_recommendation_df.sort_values(by='probability', ascending=False)
                 .groupby('product_key').head(K_PRECISE))

    # тип данных с int64 на int для БД
    result_df['product_id'] = result_df['product_id'].astype(int)

    # список словарей для передачи в БД
    result_list = result_df[['product_key', 'product_id']].to_dict(orient='records')
    
    return result_list



if __name__ == "__main__":
# Check if an argument is passed
    if len(sys.argv) > 1:
        dealer_price_path = sys.argv[1]
        with open(dealer_price_path, 'r') as json_file: 
            dealer_price = json.load(json_file)
        products_path = sys.argv[2]
        with open(products_path, 'r') as json_file: 
            products = json.load(json_file)
        result = matching(dealer_price, products)
        print(result)
    else:
        print("No files  provided.")