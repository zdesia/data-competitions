import pandas as pd
import copy
import numpy as np
from datetime import timedelta
from category_encoders import TargetEncoder
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, RobustScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import json
from itertools import product, combinations
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle
import joblib



RANDOM_STATE = 42

sales = pd.read_csv('sales_df_train.csv',  parse_dates=['date'])
categories = pd.read_csv('pr_df.csv') 
shopes = pd.read_csv('st_df.csv') 
holiday = pd.read_csv('holidays_calendar.csv', parse_dates=['date'])


def train(data_hierarchy: dict, data_stores: dict, data_train: dict,  holiday_table: dict):
    
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
        """Функция создаёт шаблон датасаета с полными временными рядами за весь период для всех комбинаций Магазин-товар из data_train
        и признаками(без лаговых).
        Параметр dates -  набор всех дат за которые нужно сгенерировать датасет со всеми комбинациями магазин-товар,
        для обучающего датасета - период дат train_data, для тестового датасета - период дат на которые нужно дать прогноз"""

        # Создадим шаблон  по всем существующим комбинациям категорий:
        # все комбинации магазин-товар из трейна
        st_pr_ids = data_train[["st_id", "pr_sku_id"]].drop_duplicates()
        # функция генерирует датасет со всеми комбинациями магазин-това-даты последовательно по датам
        def create_submissions(ids, dates):
            dataframe = copy.deepcopy(ids)
            dataframe["date"] = dates[0]
            for date in dates[1:]:
                new_dataframe = copy.deepcopy(ids)
                new_dataframe["date"] = date
                dataframe = pd.concat([dataframe, new_dataframe])
            return dataframe

        # создаём шаблон с комбинацией всех пар магазин-товар из трейна и всех дат в периоде датасета с продажами data_train
        submissions_template = create_submissions(st_pr_ids, dates)
        # назначим индексом магазин-товар-дата
        submissions_template = submissions_template.groupby(
            by=["st_id", "pr_sku_id", "date"]
        ).count()
        # заготовим столбец с магазином и sku товара, будут закодированы
        submissions_template["st_enc"] = submissions_template.index.get_level_values(0)
        submissions_template["pr_sku_enc"] = submissions_template.index.get_level_values(1)

        # Фиче-инжениринг по дате:
        submissions_template["day"] = submissions_template.index.get_level_values(2).day
        submissions_template["week"] = (
            submissions_template.index.get_level_values(2).isocalendar().week.tolist()
        )
        submissions_template["day_of_week"] = submissions_template.index.get_level_values(
            2
        ).dayofweek

        # из таблицы с выходными днями добавим в таблицу по дате признак с флагом выходного дня
        submissions_template = submissions_template.join(holiday_table.set_index("date"))

        # присоединяем признаки из других таблиц
        submissions_template = submissions_template.join(
            data_hierarchy.set_index("pr_sku_id"), on="pr_sku_id"
        )
        submissions_template = submissions_template.join(
            data_stores.set_index("st_id").drop("st_is_active", axis=1), on="st_id"
        )

        return submissions_template

    # функция для создания лаговых признаков
    def make_features_lag_alternate(data):
        data_iterated = data.copy()
        # лаги
        lags = range(1, 31)
        for lag in lags:  # [1, 2, 3, 4, 5, 6, 7, 14, 21]:
            data_iterated["lag_{}".format(lag)] = data_iterated["pr_sales_in_units"].shift(
                lag
            )
        # скользяце среднее
        data_iterated["rolling_mean_{}".format(7)] = (
            data_iterated["pr_sales_in_units"].shift().rolling(7).mean()
        )

        # скользящее среднее 7 дней две недели назад

        # удаление образовавшихся пропусков
        data_iterated = data_iterated.drop(
            data_iterated.loc[
                (data_iterated.index.get_level_values(2).day <= lags[-1])
                & (
                    data_iterated.index.get_level_values(2).month
                    == data_iterated.index.get_level_values(2).min().month
                )
            ].index
        )

        return data_iterated

    # для обучающего датасета - генерируем последовательность дат периода из train_data
    train_dates = pd.date_range(
        start=data_train["date"].min(), end=data_train["date"].max(), freq="1D"
    ).tolist()

    # генерируем шаблон с датами обучающего датасета
    template_train = prepare_full_ts_df_with_features(train_dates)

    # в данных о продажах сгруппируем по Магаззин-Товар-Дата: непромо продажи суммируются с промо в одной записи
    data_train_sum_promo = data_train.groupby(by=["st_id", "pr_sku_id", "date"]).sum()
    # присоединим к шаблону с полными временными рядами сгруппированные продажи,
    # даты без продаж ззаполним нулями, генерируем лаговые признаки
    data_train_sum_promo = make_features_lag_alternate(
        template_train.join(data_train_sum_promo, how="left").fillna(0)
    )

    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=10, random_state=RANDOM_STATE)
    encoder = TargetEncoder(cols=['st_enc', 'pr_sku_enc', 'pr_group_id', 'pr_cat_id', 'pr_subcat_id', 'st_city_id', 'st_division_code'])


    cols_claster = ['pr_group_id', 'pr_cat_id', 'pr_subcat_id', 'pr_sku_enc', 'st_enc', 'st_city_id', 'st_division_code', 'day', 'week', 
                    'day_of_week', 'holiday', 'pr_uom_id', 'st_type_format_id', 'st_type_loc_id', 'st_type_size_id']
    
    # КРОСС-ВАЛИДАЦИЯ
    def get_date_offset(date, offset):
        return (date + pd.DateOffset(days=offset)).date()
    
    #перебор гиперпараметров для этапа валидации
    def train_valid_hyperparams_tuning(train_data, valid_data, models_list):
        # инициализируем кодировщик и скейлер
        encoder = TargetEncoder(cols=['st_enc', 'pr_sku_enc', 'pr_group_id', 'pr_cat_id', 'pr_subcat_id', 
                                    'st_city_id', 'st_division_code'])
        
        scaler = RobustScaler()

        best_wape = 100
        best_model = models_list[0]
        for model in models_list:
            
            features_train = train_data.drop('pr_sales_in_units', axis=1)
            target_train = train_data['pr_sales_in_units']
            features_train = encoder.fit_transform(features_train, target_train)
            features_train[features_train.columns] = scaler.fit_transform(features_train[features_train.columns]) # scaler

            kmeans.fit(features_train[cols_claster])
            features_train['cluster'] = kmeans.predict(features_train[cols_claster])

            features_valid = valid_data.drop('pr_sales_in_units', axis=1)
            target_valid = valid_data['pr_sales_in_units']
            features_valid = encoder.transform(features_valid, target_valid)
            features_valid[features_valid.columns] = scaler.transform(features_valid[features_valid.columns]) # scaler

            features_valid['cluster'] = kmeans.predict(features_valid[cols_claster])

            model.fit(features_train, target_train)
            wape = calculate_wape(target_valid, model.predict(features_valid))
            print(model, wape)
            if wape < best_wape:
                best_wape = wape
                best_model = model
        return best_model
        
    def get_models_list(params, model):
        models = []
        for params_set in ParameterGrid(params):
            #print(params_set)
            new_model = copy.deepcopy(model)
            new_model.set_params(**params_set)
            models.append(new_model)
        return models

    def calculate_wape(y_true: np.array, y_pred: np.array):
        return np.sum(np.abs(y_true-y_pred))/np.sum(np.abs(y_true))

    def get_date_at_point(date_min, date_max, split):
        return date_min + (date_max-date_min) * split
    
    # Выборка последовательно делится на 3 части: train - обучение, valid - подбор ГП, валидация выбор лучшей, test - валидация
    def train_valid_test(data, model, params, train_valid_split_size=0.75, test_range_days=14):
        """Функция для  кросс-валидации и выбора лучшей модели. Метрика качества - WAPE.
        Каждый фолд последовательно делится на 3 части: train - обучение, valid - подбор ГП, валидация выбор лучшей, test -валидация.
        На выходе - список лучших моделей на каждом фолде с метрикой wapr на тестовой выборке"""
        # первая и последняя даты датасета
        date_min = data.index.get_level_values(2).min()
        date_max = data.index.get_level_values(2).max()
        # дата, отделяющая 75% выборки
        date_point_split = get_date_at_point(date_min, date_max, 0.75).date()
        # 1 фолд - обучающая выборка
        train_data = data.loc[(slice(None), slice(None), 
                            pd.date_range(date_min, get_date_offset(date_point_split, -test_range_days)))]
        # 1 фолд - валидационная выборка: 14 дней с последней даты обучвющей выборки + 1 день
        validation_data = data.loc[(slice(None), slice(None), 
                pd.date_range((get_date_offset(date_point_split, -test_range_days) + timedelta(1)), date_point_split))]
        # инициализируем кодировщик и скейлер
        encoder = TargetEncoder(cols=['st_enc', 'pr_sku_enc', 'pr_group_id', 'pr_cat_id', 'pr_subcat_id', 
                                    'st_city_id', 'st_division_code'])
        scaler = RobustScaler()

        # словарь для записи лучшей модели на каждом фолде 
        best_models = [] 
        # дата контроля конца датасета для остановки цикла перебора фолдов
        check_date_end = date_point_split
        while(check_date_end < date_max.date()):
            # функция выберет модель с подобранными ГП на фолде
            best_model = train_valid_hyperparams_tuning(train_data, validation_data, get_models_list(params, model))
            # создадим тестовую выборку для оценки модели на фолде, 14 дней после обучающего датасета
            date_end = get_date_offset(date_point_split, test_range_days)
            test_data = data.loc[(slice(None), slice(None), (pd.date_range(date_point_split, date_end) + timedelta(1)))] 
            features_test = test_data.drop('pr_sales_in_units', axis=1)
            target_test = test_data['pr_sales_in_units']

            # выбор лучшей модели
            train_data = pd.concat([train_data, validation_data])
            features_train = train_data.drop('pr_sales_in_units', axis=1)
            target_train = train_data['pr_sales_in_units']
            features_train = encoder.fit_transform(features_train, target_train)
            features_train[features_train.columns] = scaler.fit_transform(features_train[features_train.columns]) # scaler
            
            features_train['cluster'] = kmeans.predict(features_train[cols_claster])
            features_test = encoder.transform(features_test, target_test) # обучаем на трейне, на тесте только трансформ
            
            features_test[features_test.columns] = scaler.transform(features_test[features_test.columns]) # scaler
            features_test['cluster'] = kmeans.predict(features_test[cols_claster])

            best_model.fit(features_train, target_train)

            #тренировка на новом трейне
            test_wape = calculate_wape(target_test, best_model.predict(features_test))
            best_models.append([best_model, test_wape])
            validation_data = test_data
            date_point_split = get_date_offset(date_point_split, test_range_days)
            check_date_end = get_date_offset(date_point_split, test_range_days)
            
        #     # выбрать модель с ниаименьшим wape
        #     min(best_models_lgbm, key=lambda x: x[1])[0]
            return best_models     

    
    # набор гиперпараметров LGBM
    lgbm_params = {'random_state':[42]}

   
    # кросс-валидация модель lgbm
    model_lgbm = LGBMRegressor()
    best_models_lgbm = train_valid_test(data_train_sum_promo, model_lgbm, lgbm_params, encoder)

    # выбрать из списка модель с минимальным wape
    best_model_lgbm = min(best_models_lgbm, key=lambda x: x[1])[0]

    # сохранить обученную модель с помощью joblib
    return joblib.dump(best_model_lgbm, 'lgbm_model.pkl')

train(categories, shopes, sales, holiday)
    