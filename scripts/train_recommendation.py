#!/usr/bin/env python3
"""
Training script for recommendation experiments.
Based on omni_qwen.ipynb experiment.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from any2any_trainer.utils.config import ConfigManager
from any2any_trainer.utils.logging import setup_logging, get_logger
from any2any_trainer.models.factory import load_model
from any2any_trainer.data.recommendation_dataset import (
    RecommendationDataset, 
    RecommendationCollator,
    prepare_recommendation_data
)
from any2any_trainer.training.trainer import SimpleTrainer
from torch.utils.data import DataLoader
import torch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train recommendation model")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Override dataset name from config"
    )
    parser.add_argument(
        "--user_id_field", 
        type=str,
        help="Override user ID field name from config"
    )
    parser.add_argument(
        "--item_id_field",
        type=str,
        help="Override item ID field name from config"
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    try:
        config = ConfigManager.load_config(args.config_path)
        print(f"✅ Configuration loaded from {args.config_path}")
        
        # Override parameters if provided
        if args.dataset_name:
            config.dataset_name = args.dataset_name
        if args.user_id_field:
            config.user_id_field = args.user_id_field
        if args.item_id_field:
            config.item_id_field = args.item_id_field
            
        # Validate configuration
        ConfigManager.validate_config(config)
        print("✅ Configuration passed validation")
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return 1
    
    # Setup logging
    log_file = getattr(config, 'log_file', None)
    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    setup_logging(level="INFO", rich_console=True, log_file=log_file)
    logger = get_logger(__name__)
    
    logger.info("🚀 Starting recommendation model training...")
    logger.info(f"✅ Configuration loaded from {args.config_path}")
    logger.info("✅ Configuration passed validation")
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ Device: {device}")
    logger.info(f"🔢 Number of GPUs: {torch.cuda.device_count()}")
    
    # Load data
    logger.info("📥 Loading data...")
    try:
        train_df, val_df, test_df, item2index = prepare_recommendation_data(
            dataset_name=getattr(config, 'dataset_name', 'seniichev/amazon-fashion-2023-full'),
            user_id_field=getattr(config, 'user_id_field', 'user_id'),
            item_id_field=getattr(config, 'item_id_field', 'parent_asin'),
            title_field=getattr(config, 'title_field', 'title'),
            max_history_length=getattr(config, 'max_history_length', 10),
            min_history_length=getattr(config, 'min_history_length', 2),
            random_seed=config.seed
        )
        
        # Create datasets
        train_dataset = RecommendationDataset(
            train_df, 
            item2index=item2index,
            max_history_length=getattr(config, 'max_history_length', 10),
            min_history_length=getattr(config, 'min_history_length', 2)
        )
        
        val_dataset = RecommendationDataset(
            val_df,
            item2index=item2index,
            max_history_length=getattr(config, 'max_history_length', 10),
            min_history_length=getattr(config, 'min_history_length', 2)
        )
        
        logger.info(f"📊 Train dataset: {len(train_dataset)} examples")
        logger.info(f"📊 Validation dataset: {len(val_dataset)} examples")
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        return 1
    
    # Load model
    logger.info("📥 Loading model...")
    try:
        # Set device in config
        config.device = str(device)
        
        # Set ID vocabulary size from item mapping
        if item2index:
            config.id_vocab_size = len(item2index)
            logger.info(f"📊 Item vocabulary size: {config.id_vocab_size}")
        
        model = load_model(config)
        logger.info("✅ Model loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return 1
    
    # Create data loaders
    collator = RecommendationCollator()
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.dataloader_num_workers
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.dataloader_num_workers
    )
    
    # Create trainer
    trainer = SimpleTrainer(model, None, config)
    
    # Train model
    logger.info("🎯 Starting training...")
    try:
        trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=config.num_train_epochs
        )
        logger.info("🎉 Training completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
