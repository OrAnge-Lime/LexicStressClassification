# LexicStressClassification
Lexic stress classification for russian words

Данный репозиторий содержит решение задачи постановки ударения в русских словах.

Pipeline решения:

1. Разбиение слов по гласным буквам
2. Токенизация
3. Обучение модели. Были опробованы архитектуры на основе LSTM и Transformer

Результаты обучения:

train accuracy: 0.9409375
validation accuracy: 0.8637896728515625
score: 0.76921
public score: 0.84646


Запуск решения:

``` pip install -r requirements.txt```

``` py SoundClassifierTransformer.py```
