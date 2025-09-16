.PHONY: help install test-dataset test-sizes test-model clean train setup-wandb test-logging fix-deps train-recommendation train-semantic-recommendation test-amazon-dataset test-intermediate-validation

help:
	@echo "Amazon Fashion Multi-Task Dataset - Available commands:"
	@echo "  install        - Install dependencies"
	@echo "  fix-deps       - Fix missing dependencies for W&B"
	@echo "  setup-wandb    - Setup Weights & Biases for logging"
	@echo "  test-logging   - Test logging functionality"
	@echo "  test-dataset   - Create small test dataset (100 products)"
	@echo "  test-sizes     - Test different dataset sizes"
	@echo "  test-model     - Test model loading"
	@echo "  test-amazon-dataset - Test Amazon Fashion 2023 dataset loading"
	@echo "  test-intermediate-validation - Test intermediate validation functionality"
	@echo "  test-model-saving - Test model saving functionality"
	@echo "  test-wandb     - Test W&B functionality"
	@echo "  quick-start    - Run complete quick start workflow"
	@echo "  train          - Start model training"
	@echo "  train-recommendation - Start recommendation experiment"
	@echo "  train-semantic-recommendation - Start semantic recommendation experiment"
	@echo "  train-semantic-recommendation-safe - Start safe semantic recommendation experiment"
	@echo "  clean          - Remove generated datasets"
	@echo "  help           - Show this help"

install:
	pip install -e .

fix-deps:
	python fix_dependencies.py

setup-wandb:
	python setup_wandb.py

test-logging:
	python test_logging.py

test-dataset:
	python prepare_fashion_multitask.py --max_items 100 --output_dir test_dataset

test-sizes:
	python test_dataset_sizes.py

test-amazon-dataset:
	python test_amazon_dataset.py

test-intermediate-validation:
	python test_intermediate_validation.py

test-model-saving:
	python test_model_saving.py

test-wandb:
	python test_wandb.py

train:
	python train_fashion.py

train-recommendation:
	python scripts/train_recommendation.py configs/sft/recommendation_experiment.yaml

train-semantic-recommendation:
	python scripts/train_recommendation.py configs/sft/semantic_recommendation_experiment.yaml

train-semantic-recommendation-safe:
	python scripts/train_recommendation.py configs/sft/semantic_recommendation_experiment_safe.yaml

clean:
	rm -rf test_*_dataset fashion_test_* quick_start_dataset
	rm -rf output/fashion_multitask_model
	rm -rf output/recommendation_model
	rm -rf output/semantic_recommendation_model

