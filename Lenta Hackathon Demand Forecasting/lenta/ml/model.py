
# импорт необходимых библиотек
import pandas as pd
import numpy as np
from datetime import timedelta
import copy
from category_encoders import TargetEncoder
import joblib
import json

import warnings
warnings.filterwarnings('ignore')


def forecast(data_hierarchy: dict, data_stores: dict, data_train: dict,  holiday_table: dict):
    
    # Загрузка таблиц
    data_train, data_hierarchy, data_stores, holiday_table =\
         [pd.DataFrame(data[list(data)[0]]) for data in (data_train, data_hierarchy, data_stores, holiday_table)]

    data_train = data_train.astype({'pr_sales_type_id': 'int64'})
    data_train = data_train.astype({'pr_sales_in_units': 'int32'})
    holiday_table = holiday_table.astype({'holiday': 'int64'})

    # обработка типов данных ()
    data_train['pr_sales_type_id'] = data_train['pr_sales_type_id'].replace({False: 0, True: 1})
    data_stores['st_is_active'] = data_stores['st_is_active'].replace({False: 0, True: 1})
    holiday_table['holiday'] = holiday_table['holiday'].replace({False: 0, True: 1})


    data_train.drop('id', axis=1, inplace=True)
    holiday_table.drop('id', axis=1, inplace=True)

    #  Предобработка данных
    # удалить неактивные магазины из data_train
    stores_valid_ids = data_stores[data_stores['st_is_active'] == 1]['st_id']
    data_train = data_train[data_train['st_id'].isin(stores_valid_ids)]

    # удалить некорректных данных из data_train: 
    # - отрицательные значения в целевом признаке
    data_train = data_train.drop(data_train[data_train['pr_sales_in_units'] < 0].index)
    # - 0 товаров в целевом признаке, но есть сумма выручки и 0 выручка, но есть продажи в штуках
    data_train = data_train.drop(data_train[
                        (data_train['pr_sales_in_units'] == 0) & (data_train['pr_sales_in_rub'] != 0) |
                        (data_train['pr_sales_in_units'] != 0) & (data_train['pr_sales_in_rub'] == 0)]
                                             .index)

    # удалить ненужные столбцы из data_train
    data_train = data_train.drop(['pr_promo_sales_in_units', 'pr_promo_sales_in_rub', 'pr_sales_in_rub'], axis=1)



    # функция для создания шаблона обучающей выборки и подготовки данных для прогнозирования
    def prepare_full_ts_df_with_features(dates):
        """Функция ссоздаёт шаблон датасаета с полными временными рядами за весь период для всех комбинаций Магазин-товар из data_train 
            и признаками(без лаговых). 
            Параметр dates -  набор всех дат за которые нужно сгенерировать датасет со всеми комбинациями магазин-товар,
            для обучающего датасета - период дат train_data, для тестового датасета - период дат на которые нужно дать прогноз"""

        # Создадим шаблон  по всем существующим комбинациям категорий:
        # все комбинации магазин-товар из трейна 
        st_pr_ids = data_train[['st_id', 'pr_sku_id']].drop_duplicates()
        # функция генерирует датасет со всеми комбинациями магазин-това-даты последовательно по датам
        def create_submissions(ids, dates):
            dataframe = copy.deepcopy(ids)
            dataframe['date'] = dates[0]
            for date in dates[1:]:
                new_dataframe = copy.deepcopy(ids)
                new_dataframe['date'] = date
                dataframe = pd.concat([dataframe, new_dataframe])
            return dataframe
        # создаём шаблон с комбинацией всех пар магазин-товар из трейна и всех дат в периоде датасета с продажами data_train 
        submissions_template = create_submissions(st_pr_ids, dates)
        # назначим индексом магазин-товар-дата
        submissions_template = submissions_template.groupby(by=['st_id', 'pr_sku_id', 'date']).count()
        # заготовим столбец с магазином и sku товара, будут закодированы
        submissions_template['st_enc'] = submissions_template.index.get_level_values(0)
        submissions_template['pr_sku_enc'] = submissions_template.index.get_level_values(1)

        # Фиче-инжениринг по дате: 
        submissions_template['day'] = submissions_template.index.get_level_values(2).day
        submissions_template['week'] = submissions_template.index.get_level_values(2).isocalendar().week.tolist()
        submissions_template['day_of_week'] = submissions_template.index.get_level_values(2).dayofweek    

        # из таблицы с выходными днями добавим в таблицу по дате признак с флагом выходного дня
        submissions_template = submissions_template.join(holiday_table.set_index('date'))

        # присоединяем признаки из других таблиц
        submissions_template = submissions_template.join(data_hierarchy.set_index('pr_sku_id'), on='pr_sku_id')
        submissions_template = submissions_template.join(
                                        data_stores.set_index('st_id').drop('st_is_active', axis=1), on='st_id')

        return submissions_template


    # функция для создания лаговых признаков
    def make_features_lag_alternate(data):
        data_iterated = data.copy()
        # лаги
        for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21]: 
            data_iterated['lag_{}'.format(lag)] = data_iterated['pr_sales_in_units'].shift(lag)
        # скользяце среднее
        data_iterated['rolling_mean_{}'.format(7)] =\
                            data_iterated['pr_sales_in_units'].shift().rolling(7).mean()

        # скользящее среднее 7 дней две недели назад


        # удаление образовавшихся пропусков
    #     data_iterated = data_iterated.drop(data_iterated.loc[
    #      (data_iterated.index.get_level_values(2).day<=list_of_lag[-1]) & 
    #      (data_iterated.index.get_level_values(2).month == data_iterated.index.get_level_values(2).min().month)].index)

        return data_iterated


    # для обучающего датасета - генерируем последовательность дат периода из train_data
    train_dates = pd.date_range(start=data_train['date'].min(), end=data_train['date'].max(), freq='1D').tolist()

    # генерируем шаблон с датами обучающего датасета
    template_train = prepare_full_ts_df_with_features(train_dates)

    # из полного датасета возьмём только продажи по непромо
    data_train_no_promo = data_train[data_train['pr_sales_type_id'] == 0].drop('pr_sales_type_id', axis=1)
    # в нём сделаем индексом магазин-товар-дата
    data_train_no_promo = data_train_no_promo.groupby(by=['st_id', 'pr_sku_id', 'date']).sum() 
    # # добавим столбец с флагом промо
    # data_train_promo['pr_sales_type_id'] = 0

    # к шаблону  присоединяем исторические данные о продажах из data_train только непромо продажи, пропуски заполняем нулевыми продажами
    data_train_no_promo = template_train.join(data_train_no_promo, how='left').fillna(0)

    # создаём лаговые признаки
    data_train_no_promo = make_features_lag_alternate(data_train_no_promo)


    # функция для прогнозирования
    def forecasting(data_train, model):
        """Функция прогнозирования генерирует прогноз продаж в штуках на 14 следующих дней для всех комбинаций Магазин-Товар из датасета 
           с историческими данными о продажах data_train.
           На вход принимает обучающий датасет .....
           Подготавляивает данные для передачи в модель. Делает последовательные предсказания на каждый следующий день, на каждом шаге 
           пересчитывая лаговые признаки."""
        # последовательность дат, необходимых для предсказания 14 дней
        # берём последнюю дату из data_train и от неё генерируем следующие 14 дней
        test_dates = pd.date_range(start=(data_train.index.get_level_values(2).max() + timedelta(1)), 
                                   periods=14, freq='1D').tolist()

        # шаблон со всеми комбинациями магазин-товар-дата из обучающей выбоки на 14 дат
        data_test_submission = prepare_full_ts_df_with_features(test_dates)

        # кодирование, обучаем на трейн
        encoder = TargetEncoder(cols=['st_enc', 'pr_sku_enc', 'pr_group_id', 'pr_cat_id', 'pr_subcat_id', 
                                  'st_city_id', 'st_division_code'])
        cod_cols = ['st_enc', 'pr_sku_enc', 'pr_group_id', 'pr_cat_id', 'pr_subcat_id', 'st_city_id', 'st_division_code']
        encoder.fit(data_train[cod_cols], data_train['pr_sales_in_units'])
        data_test_submission[cod_cols] = encoder.transform(data_test_submission[cod_cols])

        # данные из обучающего датасета за последние 30 дней
        pull_dates_train_data = pd.date_range(end=(test_dates[0] - timedelta(days=1)), periods=30)
        last_days_from_train = data_train.loc[data_train.index.get_level_values(2).isin(pull_dates_train_data)]
        # кодирование
        last_days_from_train[cod_cols] = encoder.transform(last_days_from_train[cod_cols])

        # присоединяем к сабмишн данные из трейна
        data_submission_test_ext = pd.concat([last_days_from_train, data_test_submission])

        # для каждой даты, для на которую нужно сделать прогнозирование
        for date in test_dates:
            # перерассчитываем лаговые признаки
            data_submission_test_ext = make_features_lag_alternate(data_submission_test_ext)
            # выделяем срез по дате из тестовой выборки (сабмишн)
            test_one_day = data_submission_test_ext[data_submission_test_ext.index.get_level_values(2)==date]
            # предсказание моделью для среза сохраняем в столбец pr_sales_in_units, округляем и пока модуль 
            test_one_day['pr_sales_in_units'] = np.abs(np.round(model.predict(
                                    test_one_day.drop('pr_sales_in_units', axis=1))))
            # предсказанные значения для даты добавляем в общий датасет
            data_submission_test_ext.loc[data_submission_test_ext.index.get_level_values(2)==date, 'pr_sales_in_units']=\
                                        test_one_day['pr_sales_in_units']

        # удаляем лишнее из сабмишн
        data_test_submission_pred = data_submission_test_ext.loc[data_test_submission.index]
        result = data_test_submission_pred['pr_sales_in_units']

        result = data_test_submission_pred['pr_sales_in_units'].reset_index().\
                                         rename(columns={'pr_sales_in_units': 'target'})
        
        result['target'] = result['target'].astype('int')
        result['date'] = result['date'].astype('str')
        
        # возвращает список словарей
        return result.to_dict(orient='records') 


    # загрузить обученную модель из файла
    model = joblib.load('./lenta/ml/lgbm_model.pkl')

    # словарь с прогнозом
    return forecasting(data_train_no_promo, model)