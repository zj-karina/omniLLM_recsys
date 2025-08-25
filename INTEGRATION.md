# Integration Guide

This document explains how to integrate the Fashion Recommendations LLM project with the Any2Any Trainer base library.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Fashion Recommendations LLM              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ prepare_fashion_multitask.py                       │   │
│  │ train_fashion.py                                   │   │
│  │ test_dataset_sizes.py                              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Any2Any Trainer                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ models.factory                                      │   │
│  │ data.dataset                                        │   │
│  │ training.trainer                                    │   │
│  │ utils.config                                        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Option 1: Using pip

```bash
# Install base library
pip install any2any-trainer

# Install this project
pip install -e .
```

### Option 2: Using Poetry

```bash
# Install base library
poetry add any2any-trainer

# Install this project
poetry install
```

### Option 3: From source

```bash
# Clone base library
git clone https://github.com/your-username/any2any_trainer.git
cd any2any_trainer
pip install -e .

# Clone this project
git clone https://github.com/your-username/omniLLM_recsys.git
cd fashion-recommendations-llm
pip install -e .
```

## Usage

### 1. Dataset Preparation

```python
from prepare_fashion_multitask import FashionMultiTaskPreparator

# Create dataset
preparator = FashionMultiTaskPreparator()
output_dir = preparator.prepare_dataset(max_items=1000)
```

### 2. Model Training

```python
from train_fashion import train_model

# Train the model
success = train_model()
```

### 3. Integration Points

The project integrates with Any2Any Trainer through:

- **Model Loading**: Uses `any2any_trainer.models.factory.load_model()`
- **Data Loading**: Uses `any2any_trainer.data.dataset.load_dataset()`
- **Training**: Uses `any2any_trainer.training.trainer.SimpleTrainer`
- **Configuration**: Uses `any2any_trainer.utils.config.ConfigManager`

## Configuration

The training configuration extends the base library's configuration system:

```yaml
# configs/sft/fashion_multitask.yaml
model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"
modalities:
  input: ["image", "text"]
  output: ["text"]

# Uses base library's auto-detection for model_type
```

## Extending the Base Library

This project demonstrates how to extend Any2Any Trainer:

1. **Domain-Specific Data**: Amazon Fashion dataset preparation
2. **Multi-Task Learning**: 5 different recommendation tasks
3. **Custom Prompts**: Fashion-specific conversation generation
4. **Specialized Training**: Recommendation-focused configuration

## Development

To develop with this project:

```bash
# Install in development mode
pip install -e .

# Create dataset
python prepare_fashion_multitask.py --max_items 100

# Train model
python train_fashion.py
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure `any2any-trainer` is installed
2. **Configuration Error**: Check YAML syntax and paths
3. **Dataset Error**: Verify dataset format matches expected structure

### Getting Help

- Check the base library documentation
- Review configuration examples
- Run integration tests
- Check error logs
