#!/usr/bin/env python3
"""
Улучшенный скрипт обучения мультимодальной модели с поддержкой max_steps.
"""

import sys
import os
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from any2any_trainer.training.improved_trainer import ImprovedSimpleTrainer
from any2any_trainer.models.multimodal import MultimodalModel
from any2any_trainer.data.dataset import load_dataset
from any2any_trainer.utils.config import ConfigManager
from any2any_trainer.utils.logging import get_logger
from torch.utils.data import DataLoader

def main():
    """Main training function."""
    logger = get_logger(__name__)
    
    # Parse arguments
    if len(sys.argv) != 2:
        print("Usage: python train_multimodal_improved.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    try:
        # Load configuration
        logger.info(f"📋 Loading configuration from {config_path}")
        config = ConfigManager.load_config(config_path)
        logger.info("✅ Configuration loaded successfully")
        
        # Load model
        logger.info("🤖 Loading model...")
        model = MultimodalModel.from_config(config)
        logger.info("✅ Model loaded successfully")
        
        # Load tokenizer
        logger.info("🔤 Loading tokenizer...")
        from any2any_trainer.models.factory import ModelFactory
        tokenizer = ModelFactory.load_tokenizer(config)
        logger.info("✅ Tokenizer loaded successfully")
        
        # Load datasets
        logger.info("📊 Loading datasets...")
        train_dataset, eval_dataset = load_dataset(config)
        logger.info(f"✅ Train dataset: {len(train_dataset)} examples")
        if eval_dataset:
            logger.info(f"✅ Eval dataset: {len(eval_dataset)} examples")
        
        # Create data loaders
        from any2any_trainer.data.collator import MultimodalCollator
        collator = MultimodalCollator(config, tokenizer)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=config.dataloader_num_workers
        )
        
        val_dataloader = None
        if eval_dataset:
            val_dataloader = DataLoader(
                eval_dataset,
                batch_size=config.per_device_train_batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=config.dataloader_num_workers
            )
        
        # Create improved trainer
        trainer = ImprovedSimpleTrainer(model, tokenizer, config)
        
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
