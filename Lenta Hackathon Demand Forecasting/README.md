
<a id='ENG'></a>
<a href="#RUS"><img src='https://img.shields.io/badge/ENG -Go to RUS description-blue'></a>

# Product demand forecasting for Lenta
`TimeSeries` `Retail` `Machine Learning`
 
Project participation in a cross-functional hackathon to develop an ML product forecasting model for Lenta store chain (October 2023)

#### Team deliverables:
A server-side application (API) that is designed to work with predictive model data through a user interface

![Project schematic diagram](https://i.ibb.co/dbNcqtk/Flowchart-Diagram.png "Project schematic diagram.")

**Links to team repositories:**  &nbsp; &nbsp; &nbsp; [User Interface design](https://www.figma.com/file/oDb87wsTRHsC8vTtINeoBL/Команда-№1-In-Flames%2C-Хакатон.-Лента?type=design&node-id=143-3273&mode=design&t=XnoAmzIit4khUqGa-0)  &nbsp; &nbsp; &nbsp; &nbsp; [Frontend](https://github.com/Jane-Doe666/lenta) &nbsp; &nbsp; &nbsp; &nbsp; [Backend](https://github.com/kvadimas/Lenta_hackathon_backend/tree/main)



## Data Science Objective
Development of a 14-day demand forecast machine learning algorithm for own manufactured goods, preparation for launch into product use

### Contents of the repository with ML solution
- [lenta-hackathon-demand-forecasting.ipynb](https://github.com/zdesia/data-competitions/blob/main/Lenta%20Hackathon%20Demand%20Forecasting/lenta-hackathon-demand-forecasting.ipynb) - Notebook with exploratory data analysis, feature engineering, model training, validation 
- [lenta/ml](https://github.com/zdesia/data-competitions/tree/main/Lenta%20Hackathon%20Demand%20Forecasting/lenta/ml): prediction script `model.py`, model retraining script `retraining.py`, trained model `lgbm_model.pkl`
- [lenta/backend](https://github.com/zdesia/data-competitions/tree/main/Lenta%20Hackathon%20Demand%20Forecasting/lenta/backend): script to run model `app.py`

###  Stack
`Python 3.11` `Statsmodels` `Pandas` `Numpy` `Lightgbm` `Docker` `FastAPI`


### Instructions for running on a local machine

Clone the repository

```git clone git@github.com:aminaadzhieva/lenta-hackathon-forecasting.git```   
```cd lenta-hackathon-forecasting```

Install the virtual environment

```python3 -m venv venv```

Start the virtual environment

```source venv/bin/activate```

Install all dependencies

```pip install -r lenta/requirements.txt```

Start the server

```uvicorn lenta.backend.app:app```

[Start the prediction system](http://127.0.0.1:8000/api/v1/forecast/custom_response_post/) &nbsp; &nbsp; &nbsp; 
[Verify data in admin](http://31.129.109.228/admin/)


## ML Solution Description and Summary

The project solved the **problem of forecasting the time series of demand for goods** of own production for 14 days ahead: **number of products sold in units** `pr_sales_in_units` for each ***SKU/good*** (2050 units in the train dataset) in each of ***10 stores***.

"Lenta" provided historical data on **sales for 1 year**, and coded product hierarchy and store information.  

The main **pattern discovery** identified from the analysis are: 
- ***Annual trend*** - decrease in average sales during the winter season October-March.
- ***Weekly seasonality*** - sales peak on Saturday, decrease on Monday.
- Several high ***peaks in demand throughout the year, mostly around the holidays***. The sharpest sales spikes are around New Year's Day and Easter. The upswing in sales begins a few days before.
- 40.6% of records refer to sales by promotions. It is possible to have simultaneous sales of a product in the same store with and without promo. 
- The data includes products with ***incomplete time series***: sold only on days around Easter, started selling six months ago.
- All meta-attributes as store and product characteristics showed an influence on average demand

Based on the available data **new features were generated:**  
- *Calendar*: day of the week, number of the month, week number, weekend flag (taken from supplementary table)
- **Lag* 1-30 days
- *Moving average* for 7 and 14 previous days
- *Clustering* by store and product characteristics
    
To make the time series of each Store-Product combination full a new dataset was created to which missing dates were filled with zero sales.

Training, validation and selection of the best set of hyperparameters is performed on **cross validation Walk Forward**: selection of hyperparameters on fold is performed on 'valid', evaluation of the best model on fold on 'test'.   
As a result, one model among the best model at each fold is selected.

 The demand prediction by the trained model is done sequentially for each following day with intermediate recalculation of lag features (the predicted value of demand on the previous day is taken into account).

 To evaluate the model, the quality metric **WAPE** was used, calculated at the Shop-Product-Date level. 

The best result in terms of quality and speed was shown by the gradient bousting model **LightGBM**:  WAPE = **0.47**. The model performs better than baseline (prediction by last known value) with a metric of 0.69.

#### Interaction 
To implement MVP, the interaction with ML-microservice is implemented as follows: the prediction script runs every night, generates predictions for all given Store-Goods combinations and saves the result to the database. The user interacts with the database.

---

<br>
<br>

<a id='RUS'></a>
<a href="#ENG"><img src='https://img.shields.io/badge/RUS -Go to ENG description-blue'></a>

# Прогнозирование спроса товаров для “Лента”
 `TimeSeries` `Retail`  `Machine Learning`
 
Проектное  участие в кросс-функциональном хакатоне по разработке ML-продукта предсказательной модели для ООО “Лента” (октябрь 2023 г.)

### Результат работы команды:
Серверное приложение (API), которое предназначено для работы с данными предсказательной модели через интерфейс пользователя

![Принципиальная схема проекта](https://i.ibb.co/dbNcqtk/Flowchart-Diagram.png "Принципиальная схема проекта.")

**Ссылки на репозитории команды:**  &nbsp; &nbsp; &nbsp; [User Interface design](https://www.figma.com/file/oDb87wsTRHsC8vTtINeoBL/Команда-№1-In-Flames%2C-Хакатон.-Лента?type=design&node-id=143-3273&mode=design&t=XnoAmzIit4khUqGa-0)  &nbsp; &nbsp; &nbsp; &nbsp; [Frontend](https://github.com/Jane-Doe666/lenta) &nbsp; &nbsp; &nbsp; &nbsp; [Backend](https://github.com/kvadimas/Lenta_hackathon_backend/tree/main)




## Задача Data Science
Разработка алгоритма прогноза спроса на 14 дней для товаров собственного производства, подготовка для запуска в продуктовое использование

### Содержание репозитория с ML-решением
- [lenta-hackathon-demand-forecasting.ipynb](https://github.com/zdesia/data-competitions/blob/main/Lenta%20Hackathon%20Demand%20Forecasting/lenta-hackathon-demand-forecasting.ipynb) - Ноутбук с исследовательским анализом данных, инжинирингом признаков, обучением модели, валидацией 
-  [lenta/ml](https://github.com/zdesia/data-competitions/tree/main/Lenta%20Hackathon%20Demand%20Forecasting/lenta/ml): скрипт прогнозирования `model.py`, скрипт переобучения модели `retraining.py`, обученная модель `lgbm_model.pkl`
- [lenta/backend](https://github.com/zdesia/data-competitions/tree/main/Lenta%20Hackathon%20Demand%20Forecasting/lenta/backend): скрипт для запуска модели `app.py`

### Стек технологий
`Python 3.11` `Statsmodels` `Pandas` `Numpy` `Lightgbm` `Docker` `FastAPI`


### Инструкция по запуску на локальной машине

Клонировать репозиторий

```git clone git@github.com:aminaadzhieva/lenta-hackathon-forecasting.git```   
```cd lenta-hackathon-forecasting```

Установить виртуальное окружение

```python3 -m venv venv```

Запустить виртуальное окружение

```source venv/bin/activate```

Установить все зависимости

```pip install -r lenta/requirements.txt```

Запустить сервер

```uvicorn lenta.backend.app:app```

[Запустить систему предсказания](http://127.0.0.1:8000/api/v1/forecast/custom_response_post/) &nbsp; &nbsp; &nbsp; 
[Проверить данные в админке](http://31.129.109.228/admin/)


## Описание ML-решения и резюме

В проекте решалась **задача прогнозирования временного ряда спроса товаров** собственного производства на 14 дней вперёд: **число проданных товаров в штуках**  `pr_sales_in_units` для каждого ***SKU/товара*** (2050 шт. в обучающей выборке) в каждом из ***10 магазинов***.

Заказчиком предоставлены исторические данные о **продажах за 1 год**, а также в закодированном виде товарная иерархия и информация о магазинах.  

Основные **закономерности**, выявленные в результате анализа: 
- ***Годовой тренд***  - спад средних продаж в зимний сезон октябрь-март.
- ***Недельная сезонность*** - пик продаж в субботу, спад в понедельник.
- В течение года несколько высоких ***пиков спроса, в основном в районе праздников***. Самые резкие подъёмы продаж в период Нового года и Пасхи. Подъем продаж начинается за несколько дней до.
- 40,6% записей относятся к продажам по промоакциям. Возможны одновременные продажи товара в одном магазине по промо и без. 
- В данных представлены продукты с ***неполными временными рядами***: продавались только в дни около Пасхи, начали продаваться полгода назад.
- Все мета-признаки как характеристики магазинов и товаров показали влияние на средний спрос

На основе имеющихся данных **сгенерированы новые признаки:**  
- *Календарные*: день недели, число месяца, номер недели, флаг выходного дня (взят из доп. таблицы)
- *Лаговые* признаки 1-30 дней
- *Скользящее среднее* за 7 и 14 предыдущих дней
- *Кластеризация* по характеристикам магазинов и товаров
    
Чтобы временные ряды каждой комбинации Магазин-Товар были полными создан новый датасет, в который добавлены отсутствующие даты с нулевыми продажами.

 Обучение, валидация и выбор лучшего набора гиперпараметров проводится на **кросс-валидации Walk Forward**: подбор гиперпараметров на фолде проводится на valid-выборке, оценка лучшей модели на фолде на test-выборке.   
В итоге выбрана одна модель среди лучших на каждом фолде.

 Предсказание спроса обученной моделью делается последовательно на каждый следующий день с промежуточным перерасчётом лаговых признаков (учитывается предсказанное значение спроса в предыдущий день).

 Для оценки модели использовалась метрика качества  **WAPE**, посчитанная на уровне Магазин-Товар-Дата. 

Лучший результат по качеству и скорости показала модель градиентного бустинга **LightGBM**.  <br>
Полученный результат: WAPE = **0,47**. Модель работает лучше, чем baseline (предсказание последним известным значением) с метрикой 0.69.


#### Взаимодействие 
Для реализации MVP взаимодействие с ML-микросервисом реализовано следующим образом: cкрипт прогнозирования запускается каждую ночь, генерирует предсказания для всех заданных комбинаций Магазин-Товар и сохраняет результат в базу данных. Пользователь взаимодействует с базой данных.
