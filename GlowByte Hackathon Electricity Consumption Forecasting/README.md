
<a id='ENG'></a>
<a href="#RUS"><img src='https://img.shields.io/badge/ENG -Go to RUS description-blue'></a>

# Forecasting hourly electricity consumption in the region
`TimeSeries` `Industry` `Machine Learning`

Hackathon *GlowByte Autumn Hack 2023* by GlowByte company (implementing BI and Big Data solutions) (October 2023)


**Objective:**  

Develop a model for predicting hourly electricity consumption in a region 


**Provided data:**

- `train_dataset.csv` - historical data on total energy consumption in Kaliningrad region for the period 2019-01-01 - 2023-03-31
     * *date* - date
     * *time* - time
     * *target* - actual consumption, in MWh;
     * *temp* - actual temperature on the specified date;
     * *temp_pred* - temperature forecast for the specified date
     * *weather_fact* - actual weather on the specified date
     * *weather_pred* - weather forecast for the specified date
- `test_dataset.csv` - historical energy consumption data for the period 2023-04-01 - 2023-07-31

### Repository contents 
- [orakul_eda_notebook.ipynb](https://github.com/zdesia/data-competitions/blob/main/GlowByte%20Hackathon%20Electricity%20Consumption%20Forecasting/orakul_eda_notebook.ipynb) - Data Analysis Notebook
- [orakul_step2_modeling.ipynb](https://github.com/zdesia/data-competitions/blob/main/GlowByte%20Hackathon%20Electricity%20Consumption%20Forecasting/orakul_step2_modeling.ipynb) - Notebook with feature engineering, modeling
- [main.py](https://github.com/zdesia/data-competitions/blob/main/GlowByte%20Hackathon%20Electricity%20Consumption%20Forecasting/main.py) - Prediction module, takes as input a path to a private test dataset, and outputs a csv-file with a prediction 


#### Stack
`Python 3.11` `Statsmodels` `Pandas` `Numpy` `Skikit-learn` `Lightgbm` `NLTK`
  
## Solution Description and Summary

**Results of exploratory data analysis:** 

- Time series of length 4.5 years, granularity 1 hour, consistent, complete and stationary.
- Seasonality:
    - annual: decrease in summer, increases in winter, a knock-out decrease on New Year's holidays
    - weekly: increase on weekdays, decrease on weekends
    - daily: increase during the day from 7 to 23, decrease at night
    - regular drops during the New Year holiday period
- There is no consistent variation from year to year, the main influencing factor is weather: the lower the temperature during the cold season, the higher the electricity consumption.   

Weather:  
- An important factor in increasing electricity consumption is ambient temperature and therefore other aspects of weather that affect temperature and its sensation: cloud cover (lighting), wind, precipitation, humidity.
- The target attribute with temperature has a negative relationship: as the temperature decreases, consumption increases. But after +20 degrees we see a break and consumption increases with increasing temperature (cooling of rooms in hot weather).
- The textual column describing the forecasted weather weather_pred is filled in manually, because of which it contains many unique values. To save the information from the trait, we encode it using CountVectorizer with pre-cleaning. As a result, we will get a number of binary traits, which are frequently occurring cleaned words with the flag of their occurrence in the description. We will separately highlight the value of precipitation probability.

**Feature Engeeniring**

One of the main tasks to achieve a higher metric was the generation and selection of new features. In addition to correlation, their influence was evaluated experimentally.

Description | Description |
|:------|:-------|
|year |Year |
|month |Month
|dayofyear | Day of the year |
|dayofmonth | The numbered day of the month |
|dayofweek |Day of week 0-6 |
|weekofyear |Number of weeks in a year |
|quarter |Quarter of the year |
|holiday | Weekend flag |
|holiday_type |Holiday flag marking a day off (2), a day before a day off (1), a working day (0) |
|cos_time |Hour of the day (0-23), expressed as a periodic function of cos |
|cos_dayofweek | Day of the week, expressed as a periodic function of cos |
|time_of_day |The sign of the daylight hours: 0 - sunrise, 1 - light, 2 - sunset, 3 - dark |
|morning | Time of day flag: 1 if morning hours are 6-11 |
|day | Time of day flag: 1 - if daytime hours 12-17 |
|night |Time of day flag: 1 - if night hours are 0-5 |
|evening | Time of day flag: 1 - if evening hours are 18-23 |
|time_of_day |Daylight sign: 0 - sunrise, 1 - light, 2 - sunset, 3 - dark|
|lag_day |Lag to the specified number of hours back: 24-168 every 2 hours |
|temp_lag | Actual temperature 24 hours ago |
|diff_hour |Target difference between the value at this hour on the previous day and its previous hour (lag24 - lag25) |
|diff_day_hours |Difference of the target between the same hour on the two previous days (lag24 - lag48) |
|rolling |The moving average of the target over the previous 7 days, taking only the same hour |
|temp_variance_24h |Dispersion for the predicted temperature over 24 hours |
|humidity |The probability of precipitation, taken from the textual description of weather_pred |
|central_h | Central heating flag: 1 on, 0 off |
|word from weather_pred |Flag for the presence of a word in the weather_pred sign |


**Modeling**

As a result, LightGBM gradient bousting performed best for solving the problem of hourly forecasting of electricity consumption in the region. Stacking from gradient boustings was taken to improve the metric. The selection of the set of models was performed on walk forward cross-validation - the best model from each fold was taken. 

**Results**

```
Quality metrics of electricity consumption predictions for each hour on the test sample:
MAE: 6.18
MAPE: 0.0145
R@2: 0.986
```
```
Quality metrics on the latent sample (tested by customer):
MAE: 5.79
MAPE: 0.0139
R@2: 0.987
```
The work was ranked 5th in the competition.

**Other Approaches**

In addition to gradient bousting, several different approaches were investigated such as linear models, SVM, RandomForest, LSTM neural network, SARIMA. The **DLinear** approach seemed to be the most promising: one linear model predicts a moving average (trend), a second linear model predicts the difference between the real value and the trend (residuals), to predict the target, we add up two results.  Despite the high quality of trend prediction (MAE=0.5), in the case of residuals prediction, no model was able to achieve a quality higher than gradient bousting on the direct task of predicting the target.

---

<br>
<br>

<a id='RUS'></a>
<a href="#ENG"><img src='https://img.shields.io/badge/RUS -Go to ENG description-blue'></a>

# Прогнозирование почасового электропотребления в регионе
`TimeSeries` `Industry` `Machine Learning`

Хакатон *GlowByte Autumn Hack 2023* от компании GlowByte (внедрение BI и Big Data решений) (октябрь 2023 г.)


**Задача:**  

Разработка модели прогнозирования электропотребления в регионе на каждый час 


**Предоставленные данные:**

- `train_dataset.csv` – исторические данные об общем потреблении энергии в Калининградской области за период 2019-01-01 - 2023-03-31
     * *date* – дата;
     * *time* – время,  время  представлено  в  диапазоне  0  –  23,  что  означает  24 часа в сутках
     * *target* – фактическое потребление на указанную дату, в МВт\*ч
     * *temp* – фактическая температура на указанную дату
     * *temp_pred* – прогноз температуры на указанную дату
     * *weather_fact* – фактическая погода на указанную дату
     * *weather_pred* – прогноз погоды на указанную дату
- `test_dataset.csv` – исторические данные о потреблении энергии за период 2023-04-01 – 2023-07-31

### Содержание репозитория 
- [orakul_eda_notebook.ipynb](https://github.com/zdesia/data-competitions/blob/main/GlowByte%20Hackathon%20Electricity%20Consumption%20Forecasting/orakul_eda_notebook.ipynb) - Ноутбук с исследовательским анализом данных
- [orakul_step2_modeling.ipynb](https://github.com/zdesia/data-competitions/blob/main/GlowByte%20Hackathon%20Electricity%20Consumption%20Forecasting/orakul_step2_modeling.ipynb) - Ноутбук с инжинирингом признаков, обучением модели, валидацией 
-  [main.py](https://github.com/zdesia/data-competitions/blob/main/GlowByte%20Hackathon%20Electricity%20Consumption%20Forecasting/main.py) - модуль для прогнозирования,  получает на вход путь к приватному тестовому датасету, а на выходе отдает csv-файл с прогнозом 


#### Стек технологий
`Python 3.11` `Statsmodels` `Pandas` `Numpy` `Skikit-learn` `Lightgbm` `NLTK`
  
## Описание решения и резюме

**Итоги исследовательского анализа данных:** 

- Временной ряд длиной 4,5 года, гранулярностью 1 час, последовательный, полный и стационарный.
- Сезонности:
    - годовая: спад летом, повышение зимой, выбивающееся понижение в новогодние праздники
    - недельная: повышение в будни, спад в выходные
    - дневная: повышение днём с 7 до 23, понижение ночью
    - закономерные выпады в  период новогодних праздников
- Последовательных изменений по годам не наблюдается, основной влияющий фактор погода: чем ниже температура в холодный период, тем выше электропотребление.   

Погода:  
- Важный фактор повышения электропотребления - температура окружающей среды и, следовательно, другие аспекты погоды, которые влияют на температуру и её ощущение: облачность (влияет на освещение), ветер, осадки, влажность.
- Целевой признак с температурой имеет отрицательную зависимость: с понижением температуры повышается потребление. Но после +20 градусов видим перелом и уже увеличение потребления с повышением температуры (охлаждение помещений в жаркую погоду)
- Текствовый столбец с описанием прогнозируемой погоды weather_pred заполняется вручную, из-за чего содержит много уникальных значений. Для сохранения информации из признака закодируем его с помощью CountVectorizer с предварительной очисткой. В результате получим ряд бинарных признаков, которые представляют собой часто встречающиеся очищенные слова с флагом их появления в описании. Отдельно выделим значение вероятности осадков.

**Feature Engeeniring**

Одной из основных задач для достижения более высокой метрики была генерация и подбор новых признаков. Помимо корреляции их влияние оценивалось экспериментально.

|Признак |Описание |
|:------|:-------|
|year	|Год	|
|month	|Месяц	|
|dayofyear	|Порядковый день года	|
|dayofmonth	|Порядковый день месяца	|
|dayofweek	|Порядковый день недели 0-6	|
|weekofyear	|Порядковый номер недели в году	|
|quarter	|Квартал года	|
|holiday	|Флаг выходного дня (сб, вс, праздники)	|
|holiday_type	|Признак, отмечающий выходной день (2), день перед выходным (1), рабочий день (0)	|
|cos_time	|Час суток (0-23), выраженный через периодическую функцию cos	|
|cos_dayofweek | День недели, выраженный через периодическую функцию cos |
|time_of_day	|Признак светового дня: 0 - восход, 1 - светло, 2 - закат, 3 - темно	|
|morning	|Флаг времени суток: 1 - если утренние часы 6-11	|
|day	|Флаг времени суток: 1 - если дневные часы 12-17	|
|night	|Флаг времени суток: 1 - если ночные часы 0-5	|
|evening	|Флаг времени суток: 1 - если вечерние часы 18-23	|
|time_of_day	|Признак светового дня: 0 - восход, 1 - светло, 2 - закат, 3 - темно|
|lag_day	|Лаги на указанное число часов назад: 24-168 каждые 2 часа	|
|temp_lag	|Значение фактической температуры 24 часа назад	|
|diff_hour	|Разница таргета между значением в этот час в предыдущий день и его предыдущим часом (lag24 - lag25)	|
|diff_day_часы	|Разница таргета между этим же часом в два предыдущих дня (lag24 - lag48)	|
|rolling	|Скользящее среднее таргета за предыдущие 7 дней, берётся только тот же час	|
|temp_variance_24h	|Дисперсия  для прогнозированной температуры за 24 часа	|
|humidity	|Значение вероятности осадков, взято из текстового описания weather_pred	|
|central_h	|Флаг центрального отопления: 1- вкл., 0 - выкл.	|
|слово из weather_pred	|Флаг наличия слова в признаке weather_pred	|


**Моделирование**

В итоге для решения задачи почасового прогнозирования электропотребления в регионе лучше всего показал себя градиентный бустинг LightGBM. Для повышения метрики взят стекинг из градиентных бустингов. Подбор набора моделей осуществлялся на кросс-валидации walk forward - взята лучшая модель с каждого фолда. 

**Результаты**

```
Оценки предсказаний электропотребления на каждый час на тестовой выборке:
MAE: 6.185
MAPE: 0.0145
R@2: 0.986
```
Модель работает лучше бейзлайна - модели с первого этапа с метрикой МАЕ = 7,56.    

```
Метрики качества на скрытой выборке заказчика:
MAE: 5.79
MAPE: 0.0139
R@2: 0.987
```
Работа заняла **5-ое место** в соревновании.

**Другие подходы**

Помимо градиентного бустинга были исследованы несколько разных подходов такие как, линейные модели, SVM, RandomForest, LSTM нейросеть, SARIMA. Наиболее перспективным выглядел подход **DLinear**: одна линейная модель предсказывает скользящее среднее (тренд), вторая линейная модель предсказывает разницу между реальным значением и трендом (остатки), для предсказания таргета складываем предсказания двух моделей. Несмотря на высокое качество предсказания тренда (MAE=0.5), в случае с предсказанием остатков ни одной моделью не удалось добиться качества выше градиентного бустинга на прямой задаче предсказания таргета.
    
