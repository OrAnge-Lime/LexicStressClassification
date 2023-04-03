# LexicStressClassification
Lexic stress classification for russian words

Данный репозиторий содержит решение задачи постановки ударения в русских словах.

Pipeline решения:

1. Разбиение слов по гласным буквам
2. Токенизация
3. Обучение модели. Были опробованы архитектуры на основе LSTM и Transformer

Запуск решения:

``` pip install -r requirements.txt```

``` py SoundClassifierTransformer.py```
