# SF_PROJECT-3
EDA / Feature Engineering / Kaggle

>Ссылка на работу: [ссылка](EDA_Project_3_model v2.ipynb)

>Ссылка на Kaggle: [ссылка](https://www.kaggle.com/code/f999145/hotels)

# Подводя итоги:

В данной работе были проведены следующие действия:
1. Работа с признаками
   1. Преобразование признаков
   2. Очистка признаков (positive_review, negative_review)
   3. Создание новых признаков 
      1. по признаку tags
      2. взвешивание positive_review, negative_review
      3. количество слов в positive_review, negative_review
   4. Заполнение пропусков данных в признаках "lat" и "lgn"
2. Подготовили данные к модели.
   1. Заполнили пропуски нулями
   2. Удалили коррелирующие признаки
3. Вычислили MAPE 12.39%


>Если удалить менее важные признаки MAPE = 12.42%


Рейтинг на Kaggle:
![](data/Pasted%20image%2020231202163631.png)
