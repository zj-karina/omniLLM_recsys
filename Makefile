.PHONY: help install test-dataset test-sizes test-model clean train

help:
	@echo "Amazon Fashion Multi-Task Dataset - Available commands:"
	@echo "  install        - Install dependencies"
	@echo "  test-dataset   - Create small test dataset (100 products)"
	@echo "  test-sizes     - Test different dataset sizes"
	@echo "  test-model     - Test model loading"
	@echo "  quick-start    - Run complete quick start workflow"
	@echo "  train          - Start model training"
	@echo "  clean          - Remove generated datasets"
	@echo "  help           - Show this help"

install:
	pip install -r requirements.txt

test-dataset:
	python prepare_fashion_multitask.py --max_items 100 --output_dir test_dataset

test-sizes:
	python test_dataset_sizes.py

train:
	python train_fashion.py

clean:
	rm -rf test_*_dataset fashion_test_* quick_start_dataset
	rm -rf output/fashion_multitask_model

