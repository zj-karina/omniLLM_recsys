"""
Training functionality for multimodal models.

Simple implementation that extends HuggingFace Trainer.
"""

from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from datasets import Dataset

from ..utils.config import TrainingConfig, ConfigManager
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MultimodalTrainer(Trainer):
    """
    Multimodal trainer that extends HuggingFace Trainer.
    
    Simple implementation for training multimodal models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[Any] = None,
        accelerator: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize the multimodal trainer.
        
        Args:
            model: The model to train
            config: Training configuration
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            data_collator: Data collator for batching
            accelerator: Accelerate accelerator (optional)
        """
        self.config = config
        self.accelerator = accelerator
        
        # Convert config to HuggingFace TrainingArguments
        training_args = ConfigManager.to_training_arguments(config)
        
        # Initialize parent Trainer
        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            **kwargs
        )
        
        logger.info("🏋️ MultimodalTrainer initialized successfully")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the training loss.
        
        Args:
            model: The model
            inputs: Model inputs
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in batch (passed by HF Trainer)
            
        Returns:
            Loss (and outputs if return_outputs=True)
        """
        # Ensure model is in training mode
        model.train()
        
        # Force all LoRA parameters to require gradients
        for name, param in model.named_parameters():
            if 'lora' in name.lower() or 'adapter' in name.lower():
                param.requires_grad_(True)
        
        # Simple implementation - just call the model
        outputs = model(**inputs)
            
        
        # Just return the loss from model outputs
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        else:
            # Fallback - compute loss manually
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            labels = inputs.get("labels")
            
            if labels is not None and logits is not None:
                # Standard causal LM loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            else:
                # Emergency fallback
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                if trainable_params:
                    loss = sum(p.sum() for p in trainable_params) * 0.0
                else:
                    loss = torch.tensor(0.0, requires_grad=True)
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Evaluate the model.
        
        Simple implementation that just computes loss.
        """
        logger.info("📊 Starting evaluation...")
        
        # Use parent's evaluate method
        eval_results = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        logger.info(f"✅ Evaluation completed. Loss: {eval_results.get('eval_loss', 'N/A')}")
        
        # Generate examples if configured
        if self.config.generate_eval_examples:
            try:
                self._generate_examples()
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate examples: {e}")
        
        return eval_results
    
    def _generate_examples(self, num_examples: int = 3):
        """
        Generate a few examples during evaluation.
        
        Simple implementation for monitoring training progress.
        """
        logger.info("🎯 Generating evaluation examples...")
        
        if not self.eval_dataset:
            logger.warning("No evaluation dataset available for example generation")
            return
        
        # Get a few examples from eval dataset
        examples = self.eval_dataset.select(range(min(num_examples, len(self.eval_dataset))))
        
        model = self.model
        model.eval()
        
        with torch.no_grad():
            for i, example in enumerate(examples):
                try:
                    # Simple text generation (assumes the model has a generate method)
                    if hasattr(model, 'generate') and "input_ids" in example:
                        input_ids = torch.tensor([example["input_ids"][:50]], device=model.device)  # First 50 tokens
                        
                        generated_ids = model.generate(
                            input_ids,
                            max_new_tokens=self.config.max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=self.tokenizer.eos_token_id if hasattr(self, 'tokenizer') else 0,
                        )
                        
                        # Decode generated text
                        if hasattr(self, 'tokenizer'):
                            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                            logger.info(f"Example {i+1}: {generated_text[:200]}...")
                        else:
                            logger.info(f"Example {i+1}: Generated {generated_ids.shape[1]} tokens")
                    
                except Exception as e:
                    logger.warning(f"Failed to generate example {i+1}: {e}")
        
        model.train()
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the trained model.
        
        Args:
            output_dir: Directory to save the model
            _internal_call: Internal flag
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        
        logger.info(f"💾 Saving model to {output_dir}")
        
        # Use parent's save method
        super().save_model(output_dir, _internal_call)
        
        # Also save the config
        try:
            import os
            from pathlib import Path
            config_path = Path(output_dir) / "training_config.yaml"
            ConfigManager.save_config(self.config, config_path)
            logger.info(f"✅ Training config saved to {config_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save training config: {e}")


class SimpleTrainer:
    """
    Even simpler trainer for basic use cases.
    
    Minimal implementation without all the HuggingFace complexity.
    """
    
    def __init__(self, model, tokenizer, config: TrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Setup logging
        self.logger = get_logger(__name__)
        
        # Move model to GPU if available
        if hasattr(config, 'device') and config.device != 'cpu':
            self.model = self.model.to(config.device)
            self.logger.info(f"🚀 Model moved to {config.device}")
        if config.log_file:
            from pathlib import Path
            log_path = Path(config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"📝 Logging to file: {config.log_file}")
        
        # Setup Weights & Biases
        self.wandb = None
        self.wandb_initialized = False
        if config.report_to == "wandb":
            try:
                import wandb
                self.wandb = wandb
                
                # Try online mode first
                try:
                    run = wandb.init(
                        project="fashion-recommendations-llm",
                        name=config.run_name or "fashion_multitask_training",
                        config=config.dict() if hasattr(config, 'dict') else config.__dict__,
                        mode="online"
                    )
                    self.wandb_initialized = True
                    self.wandb_run = run
                    self.logger.info("🔮 Weights & Biases initialized (online mode)")
                    self.logger.info(f"🌐 Dashboard URL: {run.url}")
                except Exception as online_error:
                    self.logger.warning(f"⚠️ Online mode failed: {online_error}")
                    # Fallback to offline mode
                    try:
                        run = wandb.init(
                            project="fashion-recommendations-llm",
                            name=config.run_name or "fashion_multitask_training",
                            config=config.dict() if hasattr(config, 'dict') else config.__dict__,
                            mode="offline"
                        )
                        self.wandb_initialized = True
                        self.wandb_run = run
                        self.logger.info("🔮 Weights & Biases initialized (offline mode)")
                        self.logger.info("📁 Run data saved locally. Use 'wandb sync' to upload later.")
                    except Exception as offline_error:
                        self.logger.warning(f"⚠️ Offline mode also failed: {offline_error}")
                        self.wandb_initialized = False
                        
            except ImportError:
                self.logger.warning("⚠️ wandb not installed, skipping W&B logging")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize wandb: {e}")
                self.wandb_initialized = False
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Early stopping state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = getattr(config, 'early_stopping_patience', 3)  # Stop if no improvement for N validations
        self.min_delta = getattr(config, 'early_stopping_min_delta', 0.001)  # Minimum change to qualify as improvement
        self.eval_steps = getattr(config, 'eval_steps', 1000)  # Validate every N steps
    
    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Handle different model types
        if hasattr(self.model, 'forward') and 'text' in batch and 'id_ids' in batch:
            # Recommendation model
            device = next(self.model.parameters()).device
            # Ensure all tensors are on the correct device
            if 'labels' in batch and hasattr(batch['labels'], 'to'):
                batch['labels'] = batch['labels'].to(device)
            # id_ids will be moved to device in the model forward pass
            
            outputs = self.model(
                text=batch['text'],
                id_ids=batch['id_ids'],
                labels=batch['labels']
            )
            loss = outputs['loss']
        else:
            # Standard multimodal model
            device = next(self.model.parameters()).device
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # Check for NaN loss before backward pass
        if torch.isnan(loss):
            self.logger.warning(f"⚠️ NaN loss detected at step {self.global_step}, skipping this batch")
            return float('nan')
        
        loss.backward()
        
        # Gradient clipping
        if hasattr(self.config, 'max_grad_norm') and self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        self.optimizer.step()
        
        self.global_step += 1
        
        # Log to W&B
        if self.wandb and self.wandb_initialized and self.global_step % self.config.logging_steps == 0:
            try:
                self.wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                    "train/global_step": self.global_step,
                    "train/epoch": self.epoch
                })
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to log to W&B: {e}")
                self.wandb_initialized = False
        
        return loss.item()
    
    def validate(self, val_dataloader):
        """Validate the model and return validation loss."""
        import time
        start_time = time.time()
        self.logger.info("🔍 Starting validation...")
        self.model.eval()
        total_loss = 0
        num_batches = 0
        correct = 0
        total = 0
        
        # Progress tracking
        total_val_batches = len(val_dataloader)
        self.logger.info(f"📊 Validating on {total_val_batches} batches...")
        
        # Progress reporting every 10% of batches
        report_interval = max(1, total_val_batches // 10)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                # Handle different model types
                if hasattr(self.model, 'forward') and 'text' in batch and 'id_ids' in batch:
                    # Recommendation model
                    device = next(self.model.parameters()).device
                    # Ensure all tensors are on the correct device
                    if 'labels' in batch and hasattr(batch['labels'], 'to'):
                        batch['labels'] = batch['labels'].to(device)
                    if 'id_ids' in batch:
                        batch['id_ids'] = [ids.to(device) if hasattr(ids, 'to') else ids for ids in batch['id_ids']]
                    
                    outputs = self.model(
                        text=batch['text'],
                        id_ids=batch['id_ids'],
                        labels=batch['labels']
                    )
                    loss = outputs['loss']
                    
                    # Calculate accuracy for recommendation model
                    logits = outputs['logits']
                    preds = logits.argmax(dim=-1)
                    correct += (preds == batch['labels']).sum().item()
                    total += len(batch['labels'])
                else:
                    # Standard multimodal model
                    device = next(self.model.parameters()).device
                    batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
                    
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                total_loss += loss.item()
                num_batches += 1
                
                # Progress logging every 10% of validation
                if (batch_idx + 1) % max(1, total_val_batches // 10) == 0:
                    progress = (batch_idx + 1) / total_val_batches * 100
                    current_loss = total_loss / num_batches
                    self.logger.info(f"📊 Validation progress: {progress:.1f}% - Current loss: {current_loss:.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        accuracy = correct / total if total > 0 else 0
        
        # Calculate validation time
        validation_time = time.time() - start_time
        
        self.logger.info("=" * 60)
        self.logger.info(f"📊 VALIDATION RESULTS:")
        self.logger.info(f"   Loss: {avg_loss:.4f}")
        if total > 0:
            self.logger.info(f"   Accuracy: {accuracy:.4f} ({correct}/{total})")
        self.logger.info(f"   Batches processed: {num_batches}")
        self.logger.info(f"   Time: {validation_time:.2f}s")
        self.logger.info("=" * 60)
        
        self.model.train()
        return avg_loss
    
    def train(self, train_dataloader, val_dataloader=None, num_epochs: int = 3):
        """Training loop with early stopping based on validation loss."""
        self.logger.info(f"🎯 Starting training for {num_epochs} epochs...")
        
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
                loss = self.train_step(batch)
                total_loss += loss
                num_batches += 1
                
                # Log progress
                if num_batches % self.config.logging_steps == 0:
                    self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Step {self.global_step}, Loss: {loss:.4f}")
                
                # Save checkpoint periodically
                if hasattr(self.config, 'save_steps') and self.config.save_steps > 0 and self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
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
                        self.save_model(self.config.output_dir)
                        self.logger.info(f"💾 Best model saved to {self.config.output_dir}")
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
        
        # Final save
        self.save_model(self.config.output_dir)
        
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
        """Save checkpoint during training."""
        checkpoint_dir = f"{self.config.output_dir}/checkpoint-{self.global_step}"
        self.save_model(checkpoint_dir)
        self.logger.info(f"💾 Checkpoint saved at step {self.global_step}")
        
        # Clean up old checkpoints if save_total_limit is set
        if hasattr(self.config, 'save_total_limit') and self.config.save_total_limit > 0:
            self._cleanup_old_checkpoints()
    
    def save_model(self, output_dir: str):
        """Save the model."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
        
        # Save tokenizer if available
        if hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(output_dir)
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config.dict() if hasattr(self.config, 'dict') else self.config.__dict__
        }
        torch.save(training_state, os.path.join(output_dir, "training_state.pt"))
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints to respect save_total_limit."""
        import os
        import glob
        import shutil
        
        # Find all checkpoint directories
        checkpoint_pattern = os.path.join(self.config.output_dir, "checkpoint-*")
        checkpoint_dirs = glob.glob(checkpoint_pattern)
        
        if len(checkpoint_dirs) <= self.config.save_total_limit:
            return
        
        # Sort by step number (extract step from directory name)
        def extract_step(checkpoint_dir):
            try:
                return int(checkpoint_dir.split("checkpoint-")[-1])
            except:
                return 0
        
        checkpoint_dirs.sort(key=extract_step, reverse=True)
        
        # Remove oldest checkpoints
        checkpoints_to_remove = checkpoint_dirs[self.config.save_total_limit:]
        for checkpoint_dir in checkpoints_to_remove:
            try:
                shutil.rmtree(checkpoint_dir)
                self.logger.info(f"🗑️ Removed old checkpoint: {checkpoint_dir}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to remove checkpoint {checkpoint_dir}: {e}") 