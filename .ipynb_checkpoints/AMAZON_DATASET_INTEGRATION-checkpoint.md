# Интеграция Amazon Fashion 2023 Dataset

## 🎉 Что обновлено

### 1. Новый датасет
- **Источник**: [Amazon Fashion 2023 Full](https://huggingface.co/datasets/seniichev/amazon-fashion-2023-full) на Hugging Face
- **Размер**: 2,194,432 записей
- **Пользователи**: 259,402 уникальных пользователей
- **Товары**: 709,036 уникальных товаров
- **Примеры для обучения**: 295,688

### 2. Обновленные конфигурации
- **recommendation_experiment.yaml** - базовая конфигурация с новым датасетом
- **semantic_recommendation_experiment.yaml** - semantic конфигурация с новым датасетом
- Автоматическое определение размера словаря (709,036 товаров)

### 3. Обновленная обработка данных
- **prepare_recommendation_data()** - теперь работает с Hugging Face датасетами
- Автоматическое создание item mapping из уникальных товаров
- Группировка по пользователям для создания историй покупок
- Сортировка по timestamp для правильного порядка

### 4. Новые поля данных
```python
# Структура записи:
{
    'user_id': 'AGBFYI2DDIKXC5Y4FARTYDTQBMFQ',
    'parent_asin': 'B00LOPVX74',  # ID товара
    'title': 'CHUVORA 925 Sterling Silver...',  # Название
    'rating': 5.0,  # Рейтинг
    'timestamp': 1578528394489,  # Временная метка
    'features': [...],  # Характеристики товара
    'text': 'I think this locket is really pretty...',  # Отзыв
    'images': [...]  # Изображения товара
}
```

### 5. Тестирование
- **test_amazon_dataset.py** - скрипт для тестирования загрузки датасета
- **make test-amazon-dataset** - команда для тестирования
- Проверка корректности загрузки и обработки данных

## 🚀 Как использовать

### 1. Тестирование датасета
```bash
make test-amazon-dataset
```

### 2. Запуск базового эксперимента
```bash
make train-recommendation
```

### 3. Запуск semantic эксперимента
```bash
make train-semantic-recommendation
```

### 4. Программное использование
```python
from any2any_trainer.data.recommendation_dataset import prepare_recommendation_data

# Загрузка датасета
train_df, val_df, test_df, item2index = prepare_recommendation_data(
    dataset_name="seniichev/amazon-fashion-2023-full",
    user_id_field="user_id",
    item_id_field="parent_asin",
    title_field="title",
    max_history_length=10,
    min_history_length=2
)
```

## 📊 Статистика датасета

### Общая статистика
- **Всего записей**: 2,194,432
- **Тренировочный набор**: 2,194,432 записей
- **Валидационный набор**: 196,843 записей  
- **Тестовый набор**: 109,664 записей

### После обработки
- **Уникальных пользователей**: 259,402
- **Уникальных товаров**: 709,036
- **Примеров для обучения**: 295,688
- **Средняя длина истории**: ~4.2 товара на пользователя

### Распределение по рейтингам
- **5 звезд**: ~60% записей
- **4 звезды**: ~25% записей
- **3 звезды**: ~10% записей
- **2 звезды**: ~3% записей
- **1 звезда**: ~2% записей

## 🔧 Настройка

### Параметры датасета
```yaml
# В конфигурации
dataset_name: "seniichev/amazon-fashion-2023-full"
user_id_field: "user_id"
item_id_field: "parent_asin"
title_field: "title"
max_history_length: 10
min_history_length: 2
```

### Параметры модели
```yaml
# Размер словаря обновлен автоматически
id_vocab_size: 709036  # Количество уникальных товаров
id_dim: 512            # Размерность эмбеддингов товаров
fusion_dim: 1024       # Скрытая размерность fusion head
reduced_dim: 1024      # Размерность текстовых эмбеддингов
```

## 🎯 Преимущества нового датасета

### 1. Больший размер
- В 10+ раз больше данных чем в предыдущих экспериментах
- Более разнообразные пользователи и товары
- Лучшая репрезентативность

### 2. Богатые метаданные
- Названия товаров на английском языке
- Детальные характеристики товаров
- Изображения товаров
- Тексты отзывов

### 3. Временная информация
- Timestamp для правильной сортировки истории
- Возможность анализа временных паттернов

### 4. Готовность к использованию
- Доступен через Hugging Face
- Не требует дополнительной подготовки
- Автоматическое кэширование

## 📚 Ссылки

- [Amazon Fashion 2023 Full Dataset](https://huggingface.co/datasets/seniichev/amazon-fashion-2023-full)
- [Hugging Face Datasets Library](https://huggingface.co/docs/datasets/)
- [Original Recommendation Experiments](RECOMMENDATION_EXPERIMENTS.md)


