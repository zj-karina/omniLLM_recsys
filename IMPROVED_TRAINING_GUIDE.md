# Руководство по улучшенному обучению

## 🎯 Что исправлено

### ✅ **Для всех моделей:**
- Удален `max_steps` из конфигураций (не поддерживался `SimpleTrainer`)
- Улучшена частота сохранения чекпоинтов
- Добавлена обработка ошибок сохранения

### ✅ **Для Recommendation моделей:**
- Создан `ImprovedSimpleTrainer` с поддержкой `max_steps`
- Создан `scripts/train_recommendation_improved.py`

### ✅ **Для Fashion Multitask модели:**
- Создан `scripts/train_multimodal_improved.py`
- Использует `ImprovedSimpleTrainer` для мультимодальных моделей

## 🚀 Как использовать

### 1. **Fashion Multitask (обычная мультимодальная)**
```bash
python scripts/train_multimodal_improved.py configs/sft/fashion_multitask.yaml
```

### 2. **Recommendation (ID модальность)**
```bash
python scripts/train_recommendation_improved.py configs/sft/recommendation_experiment.yaml
```

### 3. **Semantic Recommendation (семантические ID)**
```bash
python scripts/train_recommendation_improved.py configs/sft/semantic_recommendation_experiment.yaml
```

## 🔧 Основные улучшения

### 1. **Поддержка max_steps**
Если нужно ограничить количество шагов, добавьте в конфигурацию:
```yaml
max_steps: 5000  # Ограничить до 5000 шагов
```

### 2. **Лучшая обработка ошибок**
- Продолжение обучения при ошибках в отдельных шагах
- Резервное сохранение при сбоях
- Детальное логирование ошибок

### 3. **Улучшенное сохранение**
- Более частая сохранение чекпоинтов
- Автоматическая очистка старых чекпоинтов
- Резервные копии при ошибках

## 📊 Мониторинг

### Логи
```bash
# Fashion Multitask
tail -f logs/training.log

# Recommendation
tail -f logs/recommendation_training.log

# Semantic Recommendation
tail -f logs/semantic_recommendation_safe.log
```

### Weights & Biases
- Автоматическая инициализация W&B
- Метрики в реальном времени
- Дашборд с графиками

## 🛠️ Устранение проблем

### 1. **Ошибки сохранения**
- Проверьте свободное место: `df -h`
- Используйте `save_total_limit` для ограничения чекпоинтов
- Резервные копии сохраняются в `*_backup` папках

### 2. **Ошибки обучения**
- Проверьте логи для деталей
- Обучение продолжается при ошибках в отдельных шагах
- Используйте меньший `batch_size` при нехватке памяти

### 3. **W&B проблемы**
- Проверьте подключение к интернету
- Убедитесь, что вы залогинены: `wandb login`
- При проблемах W&B переключается в offline режим

## 📁 Структура файлов

```
omniLLM_recsys/
├── configs/sft/
│   ├── fashion_multitask.yaml              # ✅ Исправлен
│   ├── recommendation_experiment.yaml      # ✅ Исправлен  
│   └── semantic_recommendation_experiment.yaml  # ✅ Исправлен
├── src/any2any_trainer/training/
│   └── improved_trainer.py                 # ✅ Новый улучшенный тренер
├── scripts/
│   ├── train_multimodal_improved.py        # ✅ Новый для мультимодальных
│   └── train_recommendation_improved.py    # ✅ Новый для рекомендаций
└── fix_training_issues.py                  # ✅ Скрипт исправления
```

## ✅ Статус моделей

Все 3 модели успешно загружены на Hugging Face Hub:
- [zjkarina/omniRecsysLLM_fasion](https://huggingface.co/zjkarina/omniRecsysLLM_fasion)
- [zjkarina/omniRecsysLLM_idmodality](https://huggingface.co/zjkarina/omniRecsysLLM_idmodality)  
- [zjkarina/omniRecsysLLM_semanticIDsmodality](https://huggingface.co/zjkarina/omniRecsysLLM_semanticIDsmodality)
