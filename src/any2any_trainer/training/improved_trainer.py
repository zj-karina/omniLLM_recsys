#!/usr/bin/env python3
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
