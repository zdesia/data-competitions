# Модуль main.py, который получает на вход путь к приватному тестовому датасету, а на выходе отдает csv-файл с прогнозом 
# в формате datetime, predict 


import argparse
import pandas as pd
import numpy as np
import re
import nltk
import joblib
import datetime

def main(hidden_data):

    hidden_data = pd.read_csv(hidden_data, parse_dates=['date'])
    hidden_data = feature_engineering_func(hidden_data)

    hidden_data = hidden_data[hidden_data.date.dt.date >= datetime.date(2023, 8, 1)]
    dates = hidden_data.date
    hours = hidden_data.time

    features = hidden_data.drop(['target', 'date'], axis=1) 

    model = joblib.load('pipe.pkl')
    
    prediction = model.predict(features)

    resampled_prediction = resample_prediction(prediction, hidden_data.index, dates, hours)
    resampled_prediction.to_csv('prediction.csv')
    return resampled_prediction.to_csv()

def create_features(data):
    data = data.copy()
    data['dayofweek'] = data.date.dt.dayofweek
    data['quarter'] = data.date.dt.quarter
    data['month'] = data.date.dt.month
    data['year'] = data.date.dt.year
    data['dayofyear'] = data.date.dt.dayofyear
    data['dayofmonth'] = data.date.dt.day
    data['weekofyear'] = data.date.dt.isocalendar().week.astype(int)

    # флаг выходного дня подтягивается из внешней таблицы по дате
    # загрузим таблицу всех выходных: викенд и праздники
    holiday_table = pd.read_csv('holidays_calendar.csv', parse_dates=['date']) 
    # добавим признак выходного дня
    data = data.merge(holiday_table[['date','holiday']], on='date', how='left')
    #  если в таблице не окажется нужной даты, заполнится наном? если наном, то для lgbm ок, если ошибка - прописать try/except
    #  data['holiday'] = data['dayofweek'].apply(lambda x: 1 if x in [5, 6] else 0)
    data['holiday_type'] = 0
    # Определяем индексы строк, где есть праздники:
    holiday_indices = data[data['holiday'] == 1].index
    # Обновляем значение в столбце 'holiday_type' на 2 для выходных/праздников
    data.loc[holiday_indices, 'holiday_type'] = 2
    # Создадим маску, чтобы найти места, где 'holiday_type' равно 2 и предыдущее значение равно 0
    mask = (data['holiday_type'] == 0) & (data['holiday_type'].shift(-24) == 2)
    # Установим предыдущее значение в 1 для отфильтрованных строк
    data.loc[mask, 'holiday_type'] = 1
    
    # час через периодическую функцию косинус
    data['cos_time'] = data['time'].apply(lambda x: np.cos(((x +1)/24) * 2 * np.pi))
    # неделя через косинус
    data['cos_dayofweek'] = data['dayofweek'].apply(lambda x: np.cos(((x)/7) * 2 * np.pi))
#     data['sin_dayofweek'] = data['dayofweek'].apply(lambda x: np.sin(((x +1)/7) * 2 * np.pi))
#     data['sin_time'] = data['time'].apply(lambda x: np.sin(((x +1)/24) * 2 * np.pi))

    # Ухудшает метрику - флаг темно, восход, светло, закат:
    sun_data = pd.read_json('sun_data.json')
    sun_data.drop('longitude', axis=1, inplace=True)
    sun_data['sunrise'] = pd.to_datetime(sun_data['sunrise'], format='%H:%M')
    sun_data['sunset'] = pd.to_datetime(sun_data['sunset'], format='%H:%M')
    # Рассчитываем продолжительность светового дня в часах, так как в оригинальном столбце ошибки:
    sun_data['daylight'] = (sun_data['sunset'] - sun_data['sunrise']).dt.total_seconds() / 3600
    sun_data['sunrise'] = sun_data['sunrise'].dt.strftime('%H.%M').astype(float).round().astype(int)
    sun_data['sunset'] = sun_data['sunset'].dt.strftime('%H.%M').astype(float).round().astype(int)
    data = data.merge(sun_data, on='date', how='left')
    # Задаем функцию для определения времени суток
    def determine_time_of_day(row):
        if row['time'] == row['sunrise']:
            return 0  # Восход
        elif row['sunrise'] <= row['time'] < row['sunset']:
            return 1  # Светло
        elif row['sunset'] == row['time']:
            return 2  # Закат
        else:
            return 3  # Темно
    data['time_of_day'] = data.apply(lambda row: determine_time_of_day(row), axis=1)
    # Удаляем ненужные столбцы -  с ними метрика еще хуже
    data.drop(['daylight', 'sunrise', 'sunset'], axis=1, inplace=True)  
    # дисперсия
    data['temp_variance_24h'] = data['temp'].rolling(window=24).var()

    def below_eight_degrees_week(temp_series):
        week_temps = temp_series.rolling(window=7*24).mean()
        return (week_temps < 8).astype(int)

    data['central_h'] = below_eight_degrees_week(data['temp_pred'])
    mask = (data['date'] >= '2019-01-01') & (data['date'] <= '2019-01-07')
    data.loc[mask, 'central_h'] = 1
    # ухудшают метрику
    # время суток
    data['morning'] = ((data['time'] >= 6) & (data['time'] < 12)).astype(int)
    data['day'] = ((data['time'] >= 12) & (data['time'] < 18)).astype(int)
    data['evening'] = ((data['time'] >= 18) & (data['time'] <= 23)).astype(int)
    data['night'] = ((data['time'] >= 0) & (data['time'] < 6)).astype(int)
    return data

def humidity_from_text_weather(dataset):
    data = dataset.copy()
    # оставляем только цифры, переводим в float    
    data['humidity'] = data['weather_pred'].fillna('').apply(lambda text: re.sub(r'[^0-9]', ' ', text))
    data['humidity'] = pd.to_numeric(data['humidity'], errors='coerce')
    data.loc[data['humidity']>100, 'humidity'] = np.nan
    return data

def preprocess_text(text):
    # перевести текст в нижний регистр
    text = text.lower()
    # заменить ё на е
    text.replace('ё', 'е')
    # оставляем только буквы 
    text = re.sub(r'[^а-я]', ' ', text)
    # Токенизация 
    text = nltk.word_tokenize(text)
    # Соединяем в текст 
    text = " ".join(text)
    # осттавить слова, содержащие не менее 3 символов
    text = re.sub(r'\b\w{1,2}\b', '', text)
    return text

# векторизация, записываем каждое слово weather_pred как отдельный столбец (наиболее часто встречающиеся слова)
def vectorize_weather_pred(data_test, count_vectorizer):
    data = data_test.copy()
    # очистили текст
    data['weather_pred'] = data['weather_pred'].fillna('').apply(lambda x: preprocess_text(x))

    # создаст матрицу: столбцы слова, значения бинарные
    text_vector = count_vectorizer.transform(data['weather_pred'])
    # преобразуем в датафрейм
    text_df = pd.DataFrame(text_vector.toarray(), columns=count_vectorizer.get_feature_names_out())
    # присодиним к основному датасету
    data = pd.concat([data, text_df], axis=1)
    # удалим признак weather_pred
    data = data.drop('weather_pred', axis=1)
    return data

def create_lag_rolling(data):
    new_data = data.copy()
    # лаги  
    for lag in range(24, 24*7, 2):
        new_data['lag_day_'+str(lag)] = new_data['target'].shift(lag)
    # лаги погоды
    new_data['temp_lag'] = new_data['temp'].shift(24)
    # скользящее среднее за предыдущие 7 дней в тот же час
    for hour in range(24):
        new_data.loc[new_data['time'] == hour, 'rolling'] =\
                        new_data.loc[new_data['time'] == hour, 'target'].shift().rolling(7).mean()
    
    # разница таргета между значением в предыдущий день и его предыдущим часом
    new_data['diff_hour'] = new_data['target'].shift(24) - new_data['target'].shift(25)
    # разница таргета между этим же часов в два предыдущих дня 
    new_data['diff_day'] = new_data['target'].shift(24) - new_data['target'].shift(48)

    # ухудшают метрику
    # разница таргета между значением в предыдущий день и его предыдущим часом
    # new_data['diff_hour'] = new_data['target'].shift(24) - new_data['target'].shift(25)
    # разница таргета между этим же часов в два предыдущих дня 
    # new_data['diff_day'] = new_data['target'].shift(24) - new_data['target'].shift(48)
    # разница таргета значением предыдущего дня этого года и предыдущего
    # new_data['diff_year'] = new_data['target'].shift(24) - new_data['target'].shift(24*365)
    for col in new_data.columns[new_data.isna().any()].tolist():
        new_data[col] = new_data[col].fillna(new_data['target'])
    new_data = new_data.fillna(99999)
    return new_data.drop(['temp', 'weather_fact'], axis=1)

def feature_engineering_func(dataset):
    df = dataset.copy()
    # замена пропусков погодных признаков предыдущим значением
    df[['temp_pred', 'weather_pred', 'weather_fact']] = df[['temp_pred', 'weather_pred', 'weather_fact']].ffill()

    # влажность
    df = humidity_from_text_weather(df)

    # загрузка count_vectorizer 
    count_vectorizer = joblib.load('count_vectorizer.pkl')
    df = vectorize_weather_pred(df, count_vectorizer)

    # календарные признаки
    df = create_features(df)

    # лаговые
    df = create_lag_rolling(df)
    
    return df

def resample_prediction(prediction, index, dates, hours):
    predicted_frame = pd.DataFrame(prediction, index = index, columns = ['predict'])
    predicted_frame['datetime'] = pd.to_datetime(dates) + pd.to_timedelta(hours, unit='h')
    predicted_frame = predicted_frame.set_index('datetime')
    return predicted_frame

def create_parser()