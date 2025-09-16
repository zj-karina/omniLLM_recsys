#!/usr/bin/env python3
"""
Simple training script for any2any multimodal models.
"""

import sys
import os
import argparse
import logging
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from any2any_trainer.utils.config import ConfigManager, TrainingConfig
from any2any_trainer.utils.logging import setup_logging, get_logger
# Временное решение проблемы с импортом factory
import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from any2any_trainer.models.factory import load_model
from any2any_trainer.data.dataset import load_dataset
from any2any_trainer.data.collator import MultimodalCollator
from any2any_trainer.training.trainer import SimpleTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train multimodal models")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only test model loading and setup without training"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration first
    try:
        config = ConfigManager.load_config(args.config_path)
        print(f"✅ Configuration loaded from {args.config_path}")
        
        # Validate configuration
        ConfigManager.validate_config(config)
        print("✅ Configuration passed validation")
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return 1
    
    # Setup logging with config
    log_file = None
    if hasattr(config, 'log_file') and config.log_file:
        log_file = config.log_file
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    setup_logging(level="INFO", rich_console=True, log_file=log_file)
    logger = get_logger(__name__)
    
    logger.info("🚀 Starting multimodal model training...")
    logger.info(f"✅ Configuration loaded from {args.config_path}")
    logger.info("✅ Configuration passed validation")
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"🔢 Number of GPUs: {torch.cuda.device_count()}")
    
    try:
        # Load model
        logger.info("📥 Loading model...")
        model = load_model(config)
        
        # Move model to device
        model = model.to(device)
        logger.info(f"✅ Model loaded and moved to {device}")
        
        # Load tokenizer 
        from any2any_trainer.models.factory import ModelFactory
        tokenizer = ModelFactory.load_tokenizer(config)
        logger.info("✅ Tokenizer loaded successfully")
        
        # Load datasets based on model type
        logger.info("📊 Loading datasets...")
        if config.model_type in ["recommendation", "semantic_recommendation"]:
            # Use recommendation dataset loading
            from any2any_trainer.data.recommendation_dataset import (
                RecommendationDataset, 
                RecommendationCollator,
                prepare_recommendation_data
            )
            
            train_df, val_df, test_df, item2index = prepare_recommendation_data(
                dataset_name=getattr(config, 'dataset_name', 'seniichev/amazon-fashion-2023-full'),
                user_id_field=getattr(config, 'user_id_field', 'user_id'),
                item_id_field=getattr(config, 'item_id_field', 'parent_asin'),
                title_field=getattr(config, 'title_field', 'title'),
                max_history_length=getattr(config, 'max_history_length', 10),
                min_history_length=getattr(config, 'min_history_length', 2),
                random_seed=getattr(config, 'seed', 42)
            )
            
            # Create datasets
            train_dataset = RecommendationDataset(
                train_df, 
                item2index=item2index,
                max_history_length=getattr(config, 'max_history_length', 10),
                min_history_length=getattr(config, 'min_history_length', 2)
            )
            
            eval_dataset = RecommendationDataset(
                val_df,
                item2index=item2index,
                max_history_length=getattr(config, 'max_history_length', 10),
                min_history_length=getattr(config, 'min_history_length', 2)
            )
            
            logger.info(f"✅ Train dataset: {len(train_dataset)} examples")
            if getattr(config, 'evaluation_strategy', 'steps') != 'no' and getattr(config, 'eval_steps', 0) > 0:
                logger.info(f"✅ Eval dataset: {len(eval_dataset)} examples")
            else:
                logger.info("📊 Eval dataset created but validation is disabled")
        else:
            # Use standard multimodal dataset loading
            train_dataset, eval_dataset = load_dataset(config)
            logger.info(f"✅ Train dataset: {len(train_dataset)} examples")
            if eval_dataset:
                logger.info(f"✅ Eval dataset: {len(eval_dataset)} examples")
        
        if args.dry_run:
            logger.info("🏁 Dry run completed successfully - stopping before training")
            return 0
        
        # Create data collator based on model type
        if config.model_type in ["recommendation", "semantic_recommendation"]:
            collator = RecommendationCollator(device=device)
        else:
            collator = MultimodalCollator(config, tokenizer)
        
        # Create data loaders
        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        # Create validation dataloader if available and validation is enabled
        val_dataloader = None
        if eval_dataset and getattr(config, 'evaluation_strategy', 'steps') != 'no' and getattr(config, 'eval_steps', 0) > 0:
            # For recommendation models, use larger batch size for validation
            if config.model_type in ["recommendation", "semantic_recommendation"]:
                # Limit validation dataset size for faster validation
                max_val_examples = 5000  # Limit to 5000 examples for faster validation
                if len(eval_dataset) > max_val_examples:
                    from torch.utils.data import Subset
                    val_indices = list(range(max_val_examples))
                    eval_dataset = Subset(eval_dataset, val_indices)
                    logger.info(f"📊 Limited validation dataset to {max_val_examples} examples for faster validation")
                
                val_batch_size = min(64, len(eval_dataset))  # Larger batch size for validation
            else:
                val_batch_size = config.per_device_eval_batch_size
            
            val_dataloader = DataLoader(
                eval_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=0
            )
            logger.info(f"✅ Validation dataloader created with {len(eval_dataset)} examples (batch_size={val_batch_size})")
        else:
            if eval_dataset:
                logger.info("📊 Validation dataset available but validation is disabled")
            else:
                logger.info("📊 No validation dataset - training without validation")
        
        # Use SimpleTrainer
        trainer = SimpleTrainer(model, tokenizer, config)
        
        # Train model with early stopping
        logger.info("🎯 Starting training with early stopping...")
        trainer.train(train_dataloader, val_dataloader, config.num_train_epochs)
        
        # Save model
        logger.info(f"💾 Saving model to {config.output_dir}")
        trainer.save_model(config.output_dir)
        
        logger.info("🎉 Training completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main()) 