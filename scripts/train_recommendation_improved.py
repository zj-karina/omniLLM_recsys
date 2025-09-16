#!/usr/bin/env python3
"""
Улучшенный скрипт обучения рекомендаций с поддержкой max_steps.
"""

import sys
import os
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from any2any_trainer.training.improved_trainer import ImprovedSimpleTrainer
from any2any_trainer.models.recommendation import RecommendationModel
from any2any_trainer.data.recommendation_dataset import prepare_recommendation_data, RecommendationCollator
from any2any_trainer.utils.config import ConfigManager
from any2any_trainer.utils.logging import get_logger
from torch.utils.data import DataLoader

def main():
    """Main training function."""
    logger = get_logger(__name__)
    
    # Parse arguments
    if len(sys.argv) != 2:
        print("Usage: python train_recommendation_improved.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    try:
        # Load configuration
        logger.info(f"📋 Loading configuration from {config_path}")
        config = ConfigManager.load_config(config_path)
        logger.info("✅ Configuration loaded successfully")
        
        # Load model
        logger.info("🤖 Loading model...")
        model = RecommendationModel.from_config(config)
        logger.info("✅ Model loaded successfully")
        
        # Load datasets
        logger.info("📊 Loading datasets...")
        train_df, val_df, test_df, item2index = prepare_recommendation_data(
            dataset_name=config.dataset_name,
            user_id_field=config.user_id_field,
            item_id_field=config.item_id_field,
            title_field=config.title_field,
            max_history_length=config.max_history_length,
            min_history_length=config.min_history_length
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
            num_workers=config.dataloader_num_workers
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=config.dataloader_num_workers
        )
        
        # Create improved trainer
        trainer = ImprovedSimpleTrainer(model, None, config)
        
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
            
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
