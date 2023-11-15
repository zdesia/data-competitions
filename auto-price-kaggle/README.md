<a id='ENG'></a>
<a href="#RUS"><img src='https://img.shields.io/badge/ENG -Go to RUS description-blue'></a>

# Predicting the price of a used car

### Project objective
Develop a ML-model to predict the price of a car for sale on the secondary market

### Data
Data from a used car selling service with their characteristics:
- training dataset with technical characteristics, equipment and price of cars, 440k objects 
- test dataset without the target feature, 110k objects

### Skills and tools
`EDA` • `Python` • `Pandas` • `Sklearn` • `Gradient Boosting` • `CatBoost` • `Linear Models` • `Clustering` • `Seaborn`

### Project Results
The best result was the CatBoostRegressor model trained on data without coding of categorical features and scaling and a filling nans with a constant.    
Kaggle test results: MAPE = 17%

Analyzing and interpreting the importance of features showed:
- The key features for price prediction are `make`, `brand_with_model`, `year`, `trim`, `odometer`.
- Less significant, but with noticeable influence `condition`, `body`, `seller`
- High values of interior color `interior` contribute to the prediction of higher price, the influence of body color `color` is less pronounced
- Insignificant contribution of `transmission`, `state`

<br>
<br>

<a id='RUS'></a>
<a href="#ENG"><img src='https://img.shields.io/badge/RUS -Go to ENG description-blue'></a>


# Предсказание стоимости поддержанного автомобиля

[**Соревнование Kaggle**](https://www.kaggle.com/competitions/used-cars-price-prediction-22ds)

### Задача
Разработать модель для предсказания стоимости продажи автомобиля на вторичном рынке.

### Данные
Данные с сервиса по продаже поддержанных автомобилей с их характеристиками:
- обучающий датасет с техническими характеристиками, комплектацией и ценой автомобилей, состоящий из  440 тыс. строк-объектов и 15 столбцов-признаков, включая целевой `sellingprice`
- аналогичный по содержанию тестовый датасет без целевого признака размером 110 тыс объектов

### Навыки и инструменты
`EDA` • `Python` • `Pandas` • `Sklearn` • `Gradient Boosting` • `CatBoost` • `Linear Models` • `Clustering` • `Seaborn`

### Результаты проекта

Лучший результат показала модель CatBoostRegressor, обученная на данных без кодирования и масштабирования с заполнением пропусков константой.   
Результаты тестировании на Kaggle: MAPE = 17% (не попал в 50%)

Анализ и интерпретация важности признаков показали:
- Ключевые признаки для предсказания цены - `make`, `brand_with_model`, `year`, `trim`, `odometer`
- Менее значимые, но с заметным влиянием `condition`, `body`, `seller`
- Высокие значения цвета интерьера `interior` вносят вклад в предсказание более высокой цены, влияние цвета кузова `color` менее выражено
- Незначительный вклад  `transmission`, `state`
