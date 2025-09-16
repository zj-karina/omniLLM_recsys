# Fashion Recommendations LLM

Multimodal Large Language Model for Amazon Fashion recommendations using [Any2Any Trainer](https://github.com/your-username/any2any_trainer).

## Overview

This project implements a multi-task recommendation system for Amazon Fashion products using:
- **Base Library**: Any2Any Trainer for multimodal model training
- **Model**: Qwen2.5-7B-Instruct with CLIP vision encoder
- **Tasks**: Product analysis, next purchase recommendations, comparisons, personalization, review generation

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Or using Poetry
poetry install

# Create test dataset
python prepare_fashion_multitask.py --max_items 100

# Train the model
python train_fashion.py
```

## Available Commands

```bash
make help              # Show all commands
make install           # Install dependencies
make setup-wandb       # Setup Weights & Biases logging
make test-dataset      # Create small test dataset
make test-sizes        # Test different dataset sizes
make train             # Start model training
make clean             # Remove generated files
```

## Logging and Monitoring

The project includes comprehensive logging and monitoring:

- **File Logging**: All training logs saved to `logs/training.log`
- **Weights & Biases**: Real-time metrics and visualizations
- **Checkpointing**: Model saved every 100 steps during training
- **Progress Tracking**: Detailed console and file output

See [LOGGING.md](LOGGING.md) for detailed setup instructions.

## Project Structure

```
fashion-recommendations-llm/
├── configs/sft/fashion_multitask.yaml  # Training configuration
├── prepare_fashion_multitask.py        # Dataset preparation script
├── train_fashion.py                    # Training script
├── test_dataset_sizes.py               # Dataset size testing
├── requirements.txt                     # Python dependencies
├── pyproject.toml                      # Poetry configuration
└── README.md                           # This file
```

## Dataset Format

Each Amazon Fashion product generates 5 task types:

1. **Product Analysis**: Analyze features and quality
2. **Next Purchase Recommendations**: Suggest similar products
3. **Product Comparison**: Compare two products
4. **Personalized Recommendations**: Based on user history
5. **Review Generation**: Simulate customer reviews

Output: JSONL files with conversations and metadata.

## Training Configuration

Key parameters in `configs/sft/fashion_multitask.yaml`:
- Model: Qwen2.5-7B-Instruct
- Vision: CLIP encoder (frozen)
- Sequence length: 4096 tokens
- Batch size: 2 (with gradient accumulation)
- Learning rate: 2e-5

## Dependencies

- **any2any-trainer**: Base multimodal training library
- **datasets**: HuggingFace datasets
- **torch**: PyTorch
- **transformers**: HuggingFace transformers
- **pandas/numpy**: Data processing
- **scikit-learn**: Machine learning utilities

## Integration with Any2Any Trainer

This project extends the base Any2Any Trainer library with:
- Fashion-specific dataset preparation
- Multi-task conversation generation
- Amazon Fashion domain knowledge
- Recommendation-specific prompts and tasks

## License

Apache-2.0 License
