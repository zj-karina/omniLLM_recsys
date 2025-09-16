# Исправления NaN Loss в обучении

## 🔧 Проблемы, которые были исправлены

### 1. Слишком высокий learning rate
- **Было**: `learning_rate: 1e-4`
- **Стало**: `learning_rate: 5e-5` (в основной конфигурации)
- **Безопасная версия**: `learning_rate: 1e-5` (в safe конфигурации)

### 2. Отсутствие gradient clipping
- **Добавлено**: `max_grad_norm: 1.0` (основная конфигурация)
- **Безопасная версия**: `max_grad_norm: 0.5` (более агрессивное clipping)

### 3. Проблемы с инициализацией весов
- **ID embeddings**: Добавлена инициализация `nn.init.normal_(weight, mean=0.0, std=0.02)`
- **Projection layers**: Добавлена инициализация `nn.init.xavier_uniform_()`
- **Fusion head**: Добавлена правильная инициализация всех слоев

### 4. Проблемы с обработкой тензоров
- **Collator**: Тензоры теперь создаются на CPU и перемещаются на GPU в модели
- **Forward pass**: Улучшена обработка device placement для id_ids
- **Gradient flow**: Добавлена проверка на NaN перед backward pass

### 5. Добавлена диагностика NaN
- **Модель**: Вывод статистики при обнаружении NaN loss
- **Тренер**: Пропуск батчей с NaN loss
- **Логирование**: Предупреждения о проблемных батчах

## 🚀 Как использовать исправления

### 1. Тестирование исправлений
```bash
python test_nan_fix.py
```

### 2. Безопасное обучение (рекомендуется)
```bash
make train-semantic-recommendation-safe
```

### 3. Обычное обучение (после тестирования)
```bash
make train-semantic-recommendation
```

## 📊 Конфигурации

### Основная конфигурация (`semantic_recommendation_experiment.yaml`)
- Learning rate: 5e-5
- Gradient clipping: 1.0
- Batch size: 2
- Mixed precision: bf16

### Безопасная конфигурация (`semantic_recommendation_experiment_safe.yaml`)
- Learning rate: 1e-5
- Gradient clipping: 0.5
- Batch size: 1
- Precision: fp32 (без mixed precision)
- Меньше шагов для тестирования

## 🔍 Диагностика

Если NaN loss все еще возникает:

1. **Проверьте логи**: Ищите сообщения "⚠️ NaN loss detected"
2. **Статистика тензоров**: Проверьте mean/std значений
3. **Используйте безопасную конфигурацию**: Более консервативные настройки
4. **Уменьшите learning rate**: Попробуйте 1e-6 или меньше

## ✅ Результаты тестирования

Все тесты прошли успешно:
- ✅ Инициализация модели
- ✅ Forward pass без NaN
- ✅ Data collator
- ✅ Gradient flow без NaN

## 📝 Дополнительные рекомендации

1. **Мониторинг**: Следите за loss в логах
2. **Checkpoints**: Сохраняйте модель чаще для восстановления
3. **Валидация**: Включите validation после стабилизации обучения
4. **Масштабирование**: Постепенно увеличивайте batch size и learning rate

