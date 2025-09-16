# Быстрый старт с логированием

## 1. Установка и настройка

```bash
# Установка зависимостей
make install

# Исправление недостающих зависимостей для W&B
make fix-deps

# Настройка Weights & Biases (опционально)
make setup-wandb

# Тестирование логирования
make test-logging
```

## 2. Подготовка данных

```bash
# Создание небольшого тестового датасета
make test-dataset

# Или создание большего датасета
python prepare_fashion_multitask.py --max_items 1000
```

## 3. Обучение с логированием

```bash
# Запуск обучения
make train
```

## 4. Мониторинг

### Файловые логи
```bash
# Просмотр логов в реальном времени
tail -f logs/training.log

# Поиск по логам
grep "ERROR" logs/training.log
grep "Checkpoint saved" logs/training.log
```

### Weights & Biases
1. Откройте https://wandb.ai/
2. Найдите проект "fashion-recommendations-llm"
3. Выберите текущий запуск

## 5. Структура файлов

```
logs/
├── training.log              # Основные логи обучения
└── test_logging.log          # Тестовые логи

output/fashion_multitask_model/
├── checkpoint-100/           # Чекпоинт на шаге 100
│   ├── model.pt
│   ├── optimizer.pt
│   ├── training_state.pt
│   └── tokenizer/
├── checkpoint-200/           # Чекпоинт на шаге 200
└── ...
```

## 6. Настройка логирования

Отредактируйте `configs/sft/fashion_multitask.yaml`:

```yaml
# Логирование
log_file: "logs/training.log"  # Файл логов
logging_steps: 10              # Частота логирования
save_steps: 100                # Частота сохранения чекпоинтов

# Weights & Biases
report_to: "wandb"             # Включить W&B
run_name: "my_training_run"    # Имя запуска
```

## 7. Устранение проблем

### Логи не создаются
```bash
# Проверьте права на запись
ls -la logs/
mkdir -p logs
```

### W&B не работает
```bash
# Проверьте подключение
wandb login
echo $WANDB_API_KEY
```

### Чекпоинты не сохраняются
```bash
# Проверьте свободное место
df -h
# Проверьте права на запись
ls -la output/
```

## 8. Примеры команд

```bash
# Полный цикл с логированием
make install && make setup-wandb && make test-dataset && make train

# Только тестирование
make test-logging

# Очистка и перезапуск
make clean && make train
```
