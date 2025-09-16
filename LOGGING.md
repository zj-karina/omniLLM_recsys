# Логирование и Мониторинг

Этот документ описывает настройку логирования и мониторинга для обучения модели.

## Файловое логирование

Все логи сохраняются в файл `logs/training.log` (настраивается в конфигурации).

### Настройка

В файле `configs/sft/fashion_multitask.yaml`:

```yaml
# Logging
log_file: "logs/training.log"  # Путь к файлу логов
logging_steps: 10              # Частота логирования (каждые 10 шагов)
```

## Weights & Biases (W&B)

Для мониторинга обучения используется Weights & Biases.

### Настройка

1. **Установка и настройка:**
   ```bash
   make setup-wandb
   ```

2. **Альтернативно - ручная настройка:**
   ```bash
   pip install wandb
   wandb login
   ```

3. **Настройка в конфигурации:**
   ```yaml
   report_to: "wandb"
   run_name: "fashion_multitask_training"
   ```

### Что логируется

- **Метрики обучения:**
  - `train/loss` - потеря на каждом шаге
  - `train/learning_rate` - скорость обучения
  - `train/global_step` - номер шага
  - `train/epoch` - номер эпохи

- **Метрики эпох:**
  - `train/epoch_loss` - средняя потеря за эпоху
  - `train/epoch` - номер эпохи

### Просмотр логов

1. **Веб-интерфейс:** https://wandb.ai/
2. **Файловые логи:** `logs/training.log`

## Промежуточное сохранение

Модель сохраняется через определенные интервалы во время обучения.

### Настройка

```yaml
save_steps: 100        # Сохранять каждые 100 шагов
save_total_limit: 3    # Хранить только 3 последних чекпоинта
```

### Структура чекпоинтов

```
output/fashion_multitask_model/
├── checkpoint-100/
│   ├── model.pt           # Веса модели
│   ├── optimizer.pt       # Состояние оптимизатора
│   ├── training_state.pt  # Состояние обучения
│   └── tokenizer/         # Токенизатор
├── checkpoint-200/
└── ...
```

## Мониторинг в реальном времени

### Консольный вывод

```
Epoch 1/3, Step 100, Loss: 2.3456
💾 Checkpoint saved at step 100
Epoch 1/3, Step 200, Loss: 2.1234
💾 Checkpoint saved at step 200
```

### Файловые логи

```
2024-01-15 10:30:15 - INFO - Epoch 1/3, Step 100, Loss: 2.3456
2024-01-15 10:30:20 - INFO - 💾 Checkpoint saved at step 100
```

## Отключение логирования

### Отключить W&B

```yaml
report_to: "none"
```

### Отключить файловое логирование

```yaml
log_file: null
```

## Устранение проблем

### W&B не работает

1. Проверьте подключение к интернету
2. Убедитесь, что вы залогинены: `wandb login`
3. Проверьте API ключ: `echo $WANDB_API_KEY`

### Файлы логов не создаются

1. Проверьте права на запись в директорию
2. Убедитесь, что путь к файлу корректен
3. Проверьте, что директория `logs/` существует

### Чекпоинты не сохраняются

1. Проверьте, что `output_dir` доступен для записи
2. Убедитесь, что `save_steps > 0`
3. Проверьте свободное место на диске

## Примеры использования

### Полный цикл обучения с логированием

```bash
# 1. Настройка
make install
make setup-wandb

# 2. Подготовка данных
python prepare_fashion_multitask.py --max_items 1000

# 3. Обучение с логированием
python train_fashion.py
```

### Просмотр логов

```bash
# Файловые логи
tail -f logs/training.log

# W&B в браузере
# Откройте https://wandb.ai/ и найдите ваш проект
```

### Восстановление из чекпоинта

```python
# Загрузка чекпоинта
checkpoint_path = "output/fashion_multitask_model/checkpoint-1000"
model.load_state_dict(torch.load(f"{checkpoint_path}/model.pt"))
```







