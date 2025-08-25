#!/usr/bin/env python3
"""
Simple training script for Amazon Fashion multi-task model.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def train_model():
    """Train the model using the existing training script."""
    print("\nStarting model training...")
    
    # Update config to use found dataset
    config_path = "configs/sft/fashion_multitask.yaml"
    
    print(f"Using config: {config_path}")
    
    # Run training
    cmd = [
        sys.executable, "scripts/train_multimodal.py",
        config_path
    ]
    
    print(f"Training command: {' '.join(cmd)}")
    print("\nNote: This will start the training process.")
    print("Make sure you have enough GPU memory and time.")
    
    response = input("\nContinue with training? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        import subprocess
        try:
            subprocess.run(cmd, check=True)
            print("Training completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Training failed: {e}")
            return False
    else:
        print("Training cancelled.")
        return False

def main():
    """Main function."""
    print("Amazon Fashion Multi-Task Model Training")
    print("=" * 40)
    
    train_model()

if __name__ == "__main__":
    main()

