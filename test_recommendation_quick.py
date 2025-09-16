#!/usr/bin/env python3
"""
Быстрый тест recommendation модели с небольшим датасетом.
"""

import sys
import os
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from any2any_trainer.training.improved_trainer import ImprovedSimpleTrainer
from any2any_trainer.models.recommendation import RecommendationModel
from any2any_trainer.data.recommendation_dataset import prepare_recommendation_data, RecommendationCollator
from any2any_trainer.utils.config import ConfigManager
from any2any_trainer.utils.logging import get_logger
from torch.utils.data import DataLoader

def main():
    """Main training function."""
    logger = get_logger(__name__)
    
    try:
        # Load configuration
        logger.info("📋 Loading configuration...")
        config = ConfigManager.load_config("configs/sft/recommendation_experiment.yaml")
        logger.info("✅ Configuration loaded successfully")
        
        # Override with smaller dataset for testing
        config.dataset_name = "seniichev/amazon-fashion-2023-full"
        config.max_history_length = 5  # Smaller history
        config.min_history_length = 2
        config.num_train_epochs = 1  # Just 1 epoch for testing
        config.save_steps = 50  # Save more frequently
        config.logging_steps = 10  # Log more frequently
        
        # Load model
        logger.info("🤖 Loading model...")
        model = RecommendationModel.from_config(config)
        logger.info("✅ Model loaded successfully")
        
        # Load datasets with limited size
        logger.info("📊 Loading datasets (limited size for testing)...")
        train_df, val_df, test_df, item2index = prepare_recommendation_data(
            dataset_name=config.dataset_name,
            user_id_field=config.user_id_field,
            item_id_field=config.item_id_field,
            title_field=config.title_field,
            max_history_length=config.max_history_length,
            min_history_length=config.min_history_length,
            max_users=1000,  # Limit to 1000 users for testing
            max_items=5000   # Limit to 5000 items for testing
        )
        
        from any2any_trainer.data.recommendation_dataset import RecommendationDataset
        
        train_dataset = RecommendationDataset(
            train_df, 
            item2index, 
            config.max_history_length, 
            config.min_history_length
        )
        
        val_dataset = RecommendationDataset(
            val_df, 
            item2index, 
            config.max_history_length, 
            config.min_history_length
        )
        
        logger.info(f"✅ Train dataset: {len(train_dataset)} examples")
        logger.info(f"✅ Val dataset: {len(val_dataset)} examples")
        
        # Create data loaders
        collator = RecommendationCollator()
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0  # No multiprocessing for testing
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0
        )
        
        # Create improved trainer
        trainer = ImprovedSimpleTrainer(model, None, config)
        
        # Train model
        logger.info("🎯 Starting training (test run)...")
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
            
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
