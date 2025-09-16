#!/usr/bin/env python3
"""
Скрипт для загрузки обученных моделей на Hugging Face Hub.
"""

import os
import shutil
import json
from pathlib import Path
from huggingface_hub import HfApi, Repository
import torch

def create_model_config(model_path: str, model_type: str) -> dict:
    """Создает config.json для модели."""
    
    if model_type == "fashion_multitask":
        config = {
            "model_type": "multimodal",
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "vision_encoder": "openai/clip-vit-large-patch14",
            "modalities": {
                "input": ["image", "text"],
                "output": ["text"]
            },
            "projection": {
                "type": "mlp",
                "hidden_size": 1024,
                "num_layers": 2,
                "dropout": 0.1
            },
            "max_seq_length": 4096,
            "special_tokens": {
                "image_start": "<img>",
                "image_end": "</img>"
            }
        }
    elif model_type == "recommendation":
        config = {
            "model_type": "recommendation",
            "base_model": "Qwen/Qwen2.5-Omni-7B",
            "id_vocab_size": 709036,
            "id_dim": 512,
            "fusion_dim": 1024,
            "reduced_dim": 1024,
            "dataset": "seniichev/amazon-fashion-2023-full",
            "max_history_length": 10,
            "min_history_length": 2
        }
    elif model_type == "semantic_recommendation":
        config = {
            "model_type": "semantic_recommendation",
            "base_model": "Qwen/Qwen2.5-Omni-7B",
            "id_vocab_size": 709036,
            "id_dim": 512,
            "fusion_dim": 1024,
            "reduced_dim": 1024,
            "vq_codebook_size": 10000,
            "vq_codebook_dim": 256,
            "use_semantic_ids": True,
            "dataset": "seniichev/amazon-fashion-2023-full",
            "max_history_length": 10,
            "min_history_length": 2
        }
    
    return config

def create_readme(model_type: str, repo_name: str) -> str:
    """Создает README.md для репозитория."""
    
    if model_type == "fashion_multitask":
        return f"""# {repo_name}

Multimodal Large Language Model для рекомендаций Amazon Fashion.

## Описание

Эта модель обучена на мультизадачном датасете Amazon Fashion и может выполнять следующие задачи:
- Анализ товаров
- Рекомендации следующих покупок
- Сравнение товаров
- Персонализированные рекомендации
- Генерация отзывов

## Архитектура

- **Базовая модель**: Qwen2.5-7B-Instruct
- **Визуальный энкодер**: CLIP-ViT-Large-Patch14
- **Проекционный слой**: MLP (1024 скрытых единиц, 2 слоя)
- **Максимальная длина последовательности**: 4096 токенов

## Использование

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Загрузка модели
model = AutoModel.from_pretrained("{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")

# Пример использования
text = "Проанализируй этот товар модной одежды"
# Добавьте код для обработки изображений и текста
```

## Обучение

Модель обучена на датасете Amazon Fashion с использованием Any2Any Trainer.

## Лицензия

Apache-2.0 License
"""
    
    elif model_type == "recommendation":
        return f"""# {repo_name}

Модель рекомендаций на основе ID модальности для Amazon Fashion.

## Описание

Эта модель использует ID эмбеддинги товаров для генерации рекомендаций на основе истории покупок пользователей.

## Архитектура

- **Базовая модель**: Qwen2.5-Omni-7B
- **Размер словаря товаров**: 709,036
- **Размерность ID эмбеддингов**: 512
- **Размерность fusion head**: 1024
- **Датасет**: Amazon Fashion 2023 Full

## Использование

```python
from any2any_trainer.models.recommendation import RecommendationModel

# Загрузка модели
model = RecommendationModel.from_pretrained("{repo_name}")

# Генерация рекомендаций
recommendations = model.predict_next_item(
    text="Пользователь купил джинсы и футболку",
    id_ids=[12345, 67890],  # ID товаров из истории
    top_k=5
)
```

## Обучение

Модель обучена на датасете Amazon Fashion 2023 с использованием ID модальности.

## Лицензия

Apache-2.0 License
"""
    
    elif model_type == "semantic_recommendation":
        return f"""# {repo_name}

Модель рекомендаций с семантическими ID для Amazon Fashion.

## Описание

Эта модель использует VQ-VAE для создания семантических ID товаров, что позволяет более точно понимать семантические связи между товарами.

## Архитектура

- **Базовая модель**: Qwen2.5-Omni-7B
- **Размер словаря товаров**: 709,036
- **Размерность ID эмбеддингов**: 512
- **VQ-VAE codebook size**: 10,000
- **VQ-VAE codebook dimension**: 256
- **Датасет**: Amazon Fashion 2023 Full

## Использование

```python
from any2any_trainer.models.recommendation import SemanticIDRecommendationModel

# Загрузка модели
model = SemanticIDRecommendationModel.from_pretrained("{repo_name}")

# Генерация рекомендаций с семантическими ID
recommendations = model.predict_next_item(
    text="Пользователь купил джинсы и футболку",
    id_ids=[12345, 67890],  # ID товаров из истории
    top_k=5,
    use_semantic_ids=True
)
```

## Обучение

Модель обучена на датасете Amazon Fashion 2023 с использованием семантических ID через VQ-VAE.

## Лицензия

Apache-2.0 License
"""

def prepare_model_for_upload(model_path: str, model_type: str, repo_name: str) -> str:
    """Подготавливает модель для загрузки."""
    
    # Создаем временную папку для модели
    temp_dir = f"temp_{model_type}_model"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Копируем файлы модели
    if os.path.exists(f"{model_path}/checkpoint-10000"):
        checkpoint_path = f"{model_path}/checkpoint-10000"
    elif os.path.exists(f"{model_path}/checkpoint-30000"):
        checkpoint_path = f"{model_path}/checkpoint-30000"
    elif os.path.exists(f"{model_path}/checkpoint-4200"):
        checkpoint_path = f"{model_path}/checkpoint-4200"
    else:
        # Ищем последний чекпоинт
        checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoint_path = f"{model_path}/{max(checkpoints, key=lambda x: int(x.split('-')[1]))}"
        else:
            raise ValueError(f"Не найден чекпоинт в {model_path}")
    
    print(f"Используем чекпоинт: {checkpoint_path}")
    
    # Копируем все файлы из чекпоинта
    for file in os.listdir(checkpoint_path):
        src = os.path.join(checkpoint_path, file)
        dst = os.path.join(temp_dir, file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    
    # Создаем config.json
    config = create_model_config(model_path, model_type)
    with open(f"{temp_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # Создаем README.md
    readme_content = create_readme(model_type, repo_name)
    with open(f"{temp_dir}/README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    return temp_dir

def upload_model(model_path: str, repo_name: str, model_type: str):
    """Загружает модель на Hugging Face Hub."""
    
    print(f"\n🚀 Подготовка модели {model_type} для загрузки...")
    
    # Подготавливаем модель
    temp_dir = prepare_model_for_upload(model_path, model_type, repo_name)
    
    print(f"📁 Временная папка создана: {temp_dir}")
    print(f"📋 Файлы в папке: {os.listdir(temp_dir)}")
    
    # Создаем репозиторий
    print(f"\n📤 Создание репозитория {repo_name}...")
    
    try:
        api = HfApi()
        
        # Создаем репозиторий если не существует
        try:
            api.create_repo(repo_id=repo_name, exist_ok=True)
            print(f"✅ Репозиторий {repo_name} создан/обновлен")
        except Exception as e:
            print(f"⚠️ Ошибка создания репозитория: {e}")
        
        # Загружаем файлы
        print(f"📤 Загрузка файлов в {repo_name}...")
        
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path):
                print(f"  Загружаем {file}...")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file,
                    repo_id=repo_name,
                    repo_type="model"
                )
        
        print(f"✅ Модель {model_type} успешно загружена в {repo_name}")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели {model_type}: {e}")
        return False
    
    finally:
        # Очищаем временную папку
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"🧹 Временная папка {temp_dir} удалена")
    
    return True

def main():
    """Основная функция."""
    
    print("🎯 ЗАГРУЗКА МОДЕЛЕЙ НА HUGGING FACE HUB")
    print("=" * 50)
    
    # Конфигурация моделей
    models_config = [
        {
            "model_path": "output/fashion_multitask_model",
            "repo_name": "zjkarina/omniRecsysLLM_fasion",
            "model_type": "fashion_multitask"
        },
        {
            "model_path": "output/recommendation_model", 
            "repo_name": "zjkarina/omniRecsysLLM_idmodality",
            "model_type": "recommendation"
        },
        {
            "model_path": "output/semantic_recommendation_model_safe",
            "repo_name": "zjkarina/omniRecsysLLM_semanticIDsmodality", 
            "model_type": "semantic_recommendation"
        }
    ]
    
    # Проверяем, что все модели существуют
    for config in models_config:
        if not os.path.exists(config["model_path"]):
            print(f"❌ Модель не найдена: {config['model_path']}")
            return
    
    print("✅ Все модели найдены")
    
    # Загружаем каждую модель
    success_count = 0
    for config in models_config:
        print(f"\n{'='*20} {config['model_type'].upper()} {'='*20}")
        
        if upload_model(
            config["model_path"],
            config["repo_name"], 
            config["model_type"]
        ):
            success_count += 1
            print(f"✅ {config['model_type']} загружена успешно")
        else:
            print(f"❌ Ошибка загрузки {config['model_type']}")
    
    print(f"\n📊 РЕЗУЛЬТАТ: {success_count}/{len(models_config)} моделей загружено")
    
    if success_count == len(models_config):
        print("\n🎉 Все модели успешно загружены на Hugging Face Hub!")
        print("\nСсылки на модели:")
        for config in models_config:
            print(f"  - https://huggingface.co/{config['repo_name']}")
    else:
        print(f"\n⚠️ Загружено {success_count} из {len(models_config)} моделей")

if __name__ == "__main__":
    main()
