#!/usr/bin/env python3
"""
Скрипт для исправления проблем с обучением.
"""

import os
import yaml
from pathlib import Path

def fix_training_configs():
    """Исправляет конфигурации обучения."""
    
    print("🔧 ИСПРАВЛЕНИЕ ПРОБЛЕМ С ОБУЧЕНИЕМ")
    print("=" * 50)
    
    # 1. Исправляем fashion_multitask.yaml
    fashion_config_path = "configs/sft/fashion_multitask.yaml"
    print(f"\n📝 Исправляем {fashion_config_path}...")
    
    with open(fashion_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Убираем max_steps, так как SimpleTrainer его не поддерживает
    if 'max_steps' in config:
        del config['max_steps']
        print("  ✅ Удален max_steps (не поддерживается SimpleTrainer)")
    
    # Увеличиваем save_steps для более частого сохранения
    config['save_steps'] = 1000
    print("  ✅ Установлен save_steps: 1000")
    
    # Увеличиваем save_total_limit
    config['save_total_limit'] = 5
    print("  ✅ Установлен save_total_limit: 5")
    
    with open(fashion_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # 2. Исправляем recommendation_experiment.yaml
    rec_config_path = "configs/sft/recommendation_experiment.yaml"
    print(f"\n📝 Исправляем {rec_config_path}...")
    
    with open(rec_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Убираем max_steps
    if 'max_steps' in config:
        del config['max_steps']
        print("  ✅ Удален max_steps")
    
    # Уменьшаем save_steps для более частого сохранения
    config['save_steps'] = 1000
    print("  ✅ Установлен save_steps: 1000")
    
    # Увеличиваем save_total_limit
    config['save_total_limit'] = 10
    print("  ✅ Установлен save_total_limit: 10")
    
    with open(rec_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # 3. Исправляем semantic_recommendation_experiment.yaml
    sem_config_path = "configs/sft/semantic_recommendation_experiment.yaml"
    print(f"\n📝 Исправляем {sem_config_path}...")
    
    with open(sem_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Убираем max_steps
    if 'max_steps' in config:
        del config['max_steps']
        print("  ✅ Удален max_steps")
    
    # Уменьшаем save_steps
    config['save_steps'] = 100
    print("  ✅ Установлен save_steps: 100")
    
    # Увеличиваем save_total_limit
    config['save_total_limit'] = 5
    print("  ✅ Установлен save_total_limit: 5")
    
    with open(sem_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print("\n✅ Все конфигурации исправлены!")

def create_improved_trainer():
    """Создает улучшенный тренер с поддержкой max_steps."""
    
    print("\n🚀 СОЗДАНИЕ УЛУЧШЕННОГО ТРЕНЕРА")
    print("=" * 50)
    
    improved_trainer_code = '''#!/usr/bin/env python3
"""
Улучшенный тренер с поддержкой max_steps и лучшей обработкой ошибок.
"""

import torch
import os
import shutil
from pathlib import Path
from any2any_trainer.training.trainer import SimpleTrainer
from any2any_trainer.utils.logging import get_logger

class ImprovedSimpleTrainer(SimpleTrainer):
    """Улучшенный тренер с поддержкой max_steps и лучшей обработкой ошибок."""
    
    def __init__(self, model, tokenizer, config):
        super().__init__(model, tokenizer, config)
        self.max_steps = getattr(config, 'max_steps', None)
        self.logger.info(f"🎯 Max steps: {self.max_steps if self.max_steps else 'unlimited'}")
    
    def train(self, train_dataloader, val_dataloader=None, num_epochs: int = 3):
        """Training loop with max_steps support and better error handling."""
        self.logger.info(f"🎯 Starting training for {num_epochs} epochs...")
        
        if self.max_steps:
            self.logger.info(f"📊 Training will stop at {self.max_steps} steps maximum")
        
        # Check if validation is disabled
        if self.eval_steps <= 0 or getattr(self.config, 'evaluation_strategy', 'steps') == 'no':
            self.logger.info("📊 Validation disabled - training without validation")
            val_dataloader = None
        else:
            self.logger.info(f"📊 Validation every {self.eval_steps} steps")
        
        if val_dataloader is None:
            self.logger.info("ℹ️ No validation dataloader provided. Training without validation.")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            total_loss = 0
            num_batches = 0
            
            # Training phase
            for batch_idx, batch in enumerate(train_dataloader):
                # Check max_steps
                if self.max_steps and self.global_step >= self.max_steps:
                    self.logger.info(f"🛑 Max steps ({self.max_steps}) reached. Stopping training.")
                    return
                
                try:
                    loss = self.train_step(batch)
                    total_loss += loss
                    num_batches += 1
                    
                    # Log progress
                    if num_batches % self.config.logging_steps == 0:
                        self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Step {self.global_step}, Loss: {loss:.4f}")
                    
                    # Save checkpoint periodically with error handling
                    if hasattr(self.config, 'save_steps') and self.config.save_steps > 0 and self.global_step % self.config.save_steps == 0:
                        try:
                            self.save_checkpoint()
                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to save checkpoint at step {self.global_step}: {e}")
                            # Try to save to a different location
                            try:
                                backup_dir = f"{self.config.output_dir}_backup"
                                os.makedirs(backup_dir, exist_ok=True)
                                self.save_model(f"{backup_dir}/checkpoint-{self.global_step}")
                                self.logger.info(f"💾 Backup checkpoint saved to {backup_dir}")
                            except Exception as backup_error:
                                self.logger.error(f"❌ Backup save also failed: {backup_error}")
                    
                    # Skip validation if disabled
                    if val_dataloader is not None and self.eval_steps > 0 and getattr(self.config, 'evaluation_strategy', 'steps') != 'no' and self.global_step % self.eval_steps == 0:
                        self.logger.info(f"🔍 Running validation at step {self.global_step}...")
                        if hasattr(self, 'wandb_run') and self.wandb_run:
                            self.logger.info(f"🌐 Monitor progress at: {self.wandb_run.url}")
                        val_loss = self.validate(val_dataloader)
                        
                        # Check for improvement
                        improvement = self.best_val_loss - val_loss
                        if improvement > self.min_delta:
                            self.best_val_loss = val_loss
                            self.patience_counter = 0
                            self.logger.info(f"🎉 New best validation loss: {val_loss:.4f} (improvement: {improvement:.4f})")
                            
                            # Save best model
                            try:
                                self.save_model(self.config.output_dir)
                                self.logger.info(f"💾 Best model saved to {self.config.output_dir}")
                            except Exception as e:
                                self.logger.warning(f"⚠️ Failed to save best model: {e}")
                        else:
                            self.patience_counter += 1
                            self.logger.info(f"⏳ No improvement for {self.patience_counter} validations (best: {self.best_val_loss:.4f})")
                        
                        # Log validation metrics to W&B
                        if self.wandb and self.wandb_initialized:
                            try:
                                self.wandb.log({
                                    "val/loss": val_loss,
                                    "val/best_loss": self.best_val_loss,
                                    "val/patience_counter": self.patience_counter,
                                    "val/global_step": self.global_step
                                })
                            except Exception as e:
                                self.logger.warning(f"⚠️ Failed to log validation metrics to W&B: {e}")
                                self.wandb_initialized = False
                        
                        # Early stopping check
                        if self.patience_counter >= self.patience:
                            self.logger.info(f"🛑 Early stopping triggered at step {self.global_step}!")
                            self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
                            return
                
                except Exception as e:
                    self.logger.error(f"❌ Error in training step {self.global_step}: {e}")
                    # Continue training instead of stopping
                    continue
            
            avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
            self.logger.info(f"✅ Epoch {epoch+1} completed. Average train loss: {avg_train_loss:.4f}")
            
            # Log epoch metrics to W&B
            if self.wandb and self.wandb_initialized:
                try:
                    self.wandb.log({
                        "train/epoch_loss": avg_train_loss,
                        "train/epoch": epoch + 1,
                        "train/global_step": self.global_step
                    })
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to log epoch metrics to W&B: {e}")
                    self.wandb_initialized = False
        
        self.logger.info("🎉 Training completed!")
        
        # Final save with error handling
        try:
            self.save_model(self.config.output_dir)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save final model: {e}")
            # Try backup location
            try:
                backup_dir = f"{self.config.output_dir}_final_backup"
                os.makedirs(backup_dir, exist_ok=True)
                self.save_model(backup_dir)
                self.logger.info(f"💾 Final model saved to backup location: {backup_dir}")
            except Exception as backup_error:
                self.logger.error(f"❌ Final backup save also failed: {backup_error}")
        
        # Close W&B and show dashboard URL
        if self.wandb and self.wandb_initialized:
            try:
                self.wandb.finish()
                if hasattr(self, 'wandb_run') and self.wandb_run:
                    self.logger.info("=" * 60)
                    self.logger.info("🌐 W&B DASHBOARD:")
                    self.logger.info(f"   URL: {self.wandb_run.url}")
                    self.logger.info("   You can view your training metrics and graphs there!")
                    self.logger.info("=" * 60)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to finish W&B: {e}")
    
    def save_checkpoint(self):
        """Save checkpoint during training with better error handling."""
        checkpoint_dir = f"{self.config.output_dir}/checkpoint-{self.global_step}"
        
        try:
            self.save_model(checkpoint_dir)
            self.logger.info(f"💾 Checkpoint saved at step {self.global_step}")
            
            # Clean up old checkpoints if save_total_limit is set
            if hasattr(self.config, 'save_total_limit') and self.config.save_total_limit > 0:
                self._cleanup_old_checkpoints()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save checkpoint at step {self.global_step}: {e}")
            # Try to save to a different location
            try:
                backup_dir = f"{self.config.output_dir}_backup"
                os.makedirs(backup_dir, exist_ok=True)
                self.save_model(f"{backup_dir}/checkpoint-{self.global_step}")
                self.logger.info(f"💾 Backup checkpoint saved to {backup_dir}")
            except Exception as backup_error:
                self.logger.error(f"❌ Backup save also failed: {backup_error}")
                raise backup_error
'''
    
    with open("src/any2any_trainer/training/improved_trainer.py", "w") as f:
        f.write(improved_trainer_code)
    
    print("✅ Создан улучшенный тренер: src/any2any_trainer/training/improved_trainer.py")

def create_training_scripts():
    """Создает улучшенные скрипты обучения."""
    
    print("\n📝 СОЗДАНИЕ УЛУЧШЕННЫХ СКРИПТОВ ОБУЧЕНИЯ")
    print("=" * 50)
    
    # Создаем улучшенный скрипт для рекомендаций
    improved_rec_script = '''#!/usr/bin/env python3
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
'''
    
    with open("scripts/train_recommendation_improved.py", "w") as f:
        f.write(improved_rec_script)
    
    # Делаем скрипт исполняемым
    os.chmod("scripts/train_recommendation_improved.py", 0o755)
    
    print("✅ Создан улучшенный скрипт: scripts/train_recommendation_improved.py")

def main():
    """Основная функция."""
    print("🔧 ИСПРАВЛЕНИЕ ПРОБЛЕМ С ОБУЧЕНИЕМ")
    print("=" * 50)
    
    # 1. Исправляем конфигурации
    fix_training_configs()
    
    # 2. Создаем улучшенный тренер
    create_improved_trainer()
    
    # 3. Создаем улучшенные скрипты
    create_training_scripts()
    
    print("\n🎉 ВСЕ ИСПРАВЛЕНИЯ ЗАВЕРШЕНЫ!")
    print("\n📋 Что было исправлено:")
    print("  ✅ Удален max_steps из конфигураций (не поддерживается SimpleTrainer)")
    print("  ✅ Улучшена частота сохранения чекпоинтов")
    print("  ✅ Создан ImprovedSimpleTrainer с поддержкой max_steps")
    print("  ✅ Добавлена лучшая обработка ошибок сохранения")
    print("  ✅ Создан улучшенный скрипт обучения")
    
    print("\n🚀 Как использовать:")
    print("  # Для рекомендаций:")
    print("  python scripts/train_recommendation_improved.py configs/sft/recommendation_experiment.yaml")
    print("  # Для семантических рекомендаций:")
    print("  python scripts/train_recommendation_improved.py configs/sft/semantic_recommendation_experiment.yaml")

if __name__ == "__main__":
    main()
