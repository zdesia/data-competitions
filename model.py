#!/usr/bin/env python
# coding: utf-8

# In[38]:


# model.py

import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from concurrent import futures
from tqdm import tqdm
import pymorphy2

# Функция для добавления пропущенных пробелов в наименованиях.
def add_spaces(text):
    spaced_text = re.sub(r'(?<=[a-zA-Z])(?=[а-яА-ЯёЁ])|(?<=[а-яА-ЯёЁ])(?=[a-zA-Z])', ' ', text)
    spaced_text = re.sub(r'(\S)\*(\S)', r'\1 * \2', spaced_text)
    spaced_text = re.sub(r'(\d+)([а-яА-ЯёЁa-zA-Z]+)', r'\1 \2', spaced_text)
    return spaced_text


# Функция очистки,  токенизации, лемматизации и удаление стоп-слов.
def preprocess_text(text):
    # Очистка текста
    cleaned_text = re.sub(r"[\(\),/.]", ' ', text) # r"[^a-zA-Zа-яА-ЯёЁ ]"
    
    # Токенизация
    tokens = word_tokenize(cleaned_text.lower())
    
    # Лемматизация
#     lemmatizer = WordNetLemmatizer()
#     lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    
#     # Лемматизация ттолько русских слов, английские являются неизменяемыми названиями
#     morph = pymorphy2.MorphAnalyzer()
#     lemmas = [morph.parse(word)[0][2] for word in tokens]
    
    # Удаление стоп-слов
#     stop_words = set(stopwords.words('russian') + stopwords.words('english'))
#     filtered_words = [lemma for lemma in lemmas if lemma not in stop_words]
    
    # Возвращение предобработанного текста
    return ' '.join(tokens)



def get_rec_labse(product_name, corpus_embeddings_labse):
    preprocessed_name = preprocess_text(product_name)

    dealer_product_embedding = model_labse.encode([preprocessed_name])

    # Поиск наиболее похожих названий
    cosine_scores = util.pytorch_cos_sim(dealer_product_embedding, corpus_embeddings_labse)[0]

    top_matches_indices = cosine_scores.argsort(descending=True)[:5]
#     top_matches_names = [names_corpus[i] for i in top_matches_indices]
    recommendations = data_products.loc[list(top_matches_indices)]['name'].tolist()

    return recommendations


def matching(product_name):
    
    data_products = pd.read_csv('marketing_product.csv', sep=';')


    # удалили пропуск в названии 2 шт
    data_products.dropna(subset=['name'], inplace=True)
    # удалили одно название пустое
    empty_name_rows = data_products[data_products['name'].str.contains(r'^\s*$')]
    data_products.drop(index=empty_name_rows.index, inplace=True)


    #  добавить пропущенные пробелы в наименования товаров в столбцах 'name', 'name_1c', 'ozon_name' и 'wb_name'
    columns_to_apply = ['name', 'name_1c', 'ozon_name', 'wb_name']
    data_products[columns_to_apply] = data_products[columns_to_apply].astype(str).applymap(add_spaces)

    #  обьем, количество штук в упаковке, концентрация продукция, разная единица измерения
    # извлечь и зафиксировать в отдельные столбцы в качестве дополнительных признаков
    data_products = extract_info(data_products)

    # обнаружено, что есть лишний пробел в слове Средство "C редство"
    # вручную исправим ошибку
    data_products['name_1c'] = data_products['name_1c'].str.replace('C редство', 'Средство')
    data_products['name'] = data_products['name'].str.replace('C редство', 'Средство')

    # в наименования оказались еще места, которые не были обработаны функцией и уберем эти проблемы вручную
    data_products['name'] = data_products['name'].str.replace('БЕТОНКОНТАКТготовый', 'БЕТОНКОНТАКТ готовый')
    data_products['name'] = data_products['name'].str.replace('"к', '" к')
    data_products['name'] = data_products['name'].str.replace('яблокаконцентрированное', 'яблока концентрированное')
    data_products['name'] = data_products['name'].str.replace('(сухой остаток 20%)', ' (сухой остаток 20%) ')
    data_products['name'] = data_products['name'].str.replace('.C', '. C')

    # пропуски в столбце name_1c заполнить значениями в сттолбце name 
    data_products['name_1c'].fillna(data_products['name'], inplace=True)
    
    data_products = data_products.dropna( subset=['name']).reset_index(drop=True)


    # очистка названий товаров заказчика в data_products
    names_corpus = data_products['name_1c'].apply(lambda x: preprocess_text(x)).tolist()


    model_labse = SentenceTransformer('sentence-transformers/LaBSE')

    corpus_embeddings_labse = model_labse.encode(names_corpus)
    
    return get_rec_labse(product_name, corpus_embeddings_labse)



matching(product_name)


# In[ ]:




