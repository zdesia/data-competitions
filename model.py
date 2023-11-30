

import pandas as pd
import numpy as np
import re
import nltk
# import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
tqdm.pandas() # для работы progress_apply в пандас

from sentence_transformers import SentenceTransformer, util
from concurrent import futures



class nice_model():
    
    def __init__(self, data_products, data_delears):
        # путь до датасетов 
        # дата фрейм с данными о продуктах,  признаки и ключи 
        #self.data_products = pd.read_csv(data_products,sep=';')[['id','name_1c']]
        self.data_products = data_products[['id','name_1c', 'name']]
        #  датафрейм с которыми будет работать модуль  
        #self.data_delears = pd.read_csv(data_delears,sep=';')[['id','product_key',
                                                               #'product_name','dealer_id']]
        self.data_delears = data_delears[['product_key',
                                          'product_name','dealer_id']]
        
        # Тут в любом случае нужно оставлять только тензоры на данных в обучении
#         with open('model_labse_14.pkl', 'rb') as file: 
#             self.model_labse= pickle.load(file)
        self.model_labse = SentenceTransformer('sentence-transformers/LaBSE')
      
   # предобработка, все что угодно можно добавить 
#     def preprocessing(self):
        
        def add_spaces(data):
            # Добавляет пробелы 
            spaced_text = re.sub(r'(?<=[a-zA-Z])(?=[а-яА-ЯёЁ])|(?<=[а-яА-ЯёЁ])(?=[a-zA-Z])', ' ', data)
            spaced_text = re.sub(r'(\S)\*(\S)', r'\1 * \2', spaced_text)
            spaced_text = re.sub(r'(\d+)([а-яА-ЯёЁa-zA-Z]+)', r'\1 \2', spaced_text)
            return spaced_text
       
         # удалили пропуск в названии 2 шт
        self.data_products = self.data_products.dropna(subset=['name'])
        # удалили одно название пустое
#         empty_name_rows = data_products[data_products['name'].str.contains(r'^\s*$')]
#         self.data_products.drop(index=empty_name_rows.index, inplace=True)
        
        self.data_products['name_1c'] = self.data_products['name_1c'].astype(str).apply(add_spaces)
        self.data_delears['product_name'] = self.data_delears['product_name'].astype(str).apply(add_spaces)
        
        #  исправим ошибку
        self.data_products['name_1c'] = self.data_products['name_1c'].str.replace('C редство', 'Средство')
        self.data_products['name'] = self.data_products['name'].str.replace('C редство', 'Средство')
        
        # пропуски в столбце name_1c заполнить значениями в сттолбце name 
        self.data_products['name_1c'] = self.data_products['name_1c'].fillna(self.data_products['name'])
        
        # удалилим пропуск в названии 
        self.data_products = self.data_products.dropna(subset=['name_1c'])
        self.data_delears = self.data_delears.dropna(subset=['product_name'])
        self.data_delears = self.data_delears
        self.data_products = self.data_products
       

    # функция реализации модели      
    def get_recomendation(self):
        
        # Корпус с названиями в ключах и id в значениях в продуктах заказчика
        names_corpus = dict(zip(self.data_products['name_1c'], self.data_products['id']))
        # Преобразование текстовых данных products в векторы с помощью lbse
        
        # Делаем корпус наших данных от заказчика. Думаю что можно сделать отдельный файл и сохранить его 
        corpus_embeddings_labse = self.model_labse.encode(list(names_corpus.keys()))
        
        # функция для работы с батчами
        def use_batch_get_rec(func_get_rec, batch_size = 2000):
      
            # Создание пула потоков # параллельные вычисления
            executor = futures.ThreadPoolExecutor()
            recommendations = []
        
            # Разделение данных на пакеты и обработка пакетов
            for i in range(0, len(self.data_delears), batch_size):
                batch = self.data_delears['product_name'].iloc[i:i+batch_size]
                batch_recommendations = list(tqdm(executor.map(get_rec_labse, batch), total=len(batch)))
                recommendations.extend(batch_recommendations)
                
            return recommendations
        
        # модель взял уже заранее сохраненную, также можно и тензоры сохранить отдельно 
        def get_rec_labse(product_name):

            # эмбендинг 
            dealer_product_embedding = self.model_labse.encode(product_name)

            # Поиск наиболее похожих названий
            cosine_scores = util.pytorch_cos_sim(dealer_product_embedding, corpus_embeddings_labse)[0]
           # print(cosine_scores)

            top_matches_indices = cosine_scores.argsort(descending=True)[:5]
            top_matches_names = [list(names_corpus.keys())[i] for i in top_matches_indices]
            id_product_rec = [names_corpus[i] for i in top_matches_names]
            
            return top_matches_names, id_product_rec
        self.data_delears[['recommendations_name','ecommendations_id']]  = use_batch_get_rec(get_rec_labse)
        return self.data_delears.to_csv('results_table.csv', index=False)
    
    # загрузка датасетов

data_dealers = pd.read_csv('marketing_dealerprice.csv', sep=";", parse_dates=['date'])
data_products = pd.read_csv('marketing_product.csv', sep=";", index_col='Unnamed: 0',)
    
model = nice_model(data_products, data_dealers.head(100))
model.get_recomendation()
    