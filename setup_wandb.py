#!/usr/bin/env python3
"""
Setup script for Weights & Biases integration.
"""

import os
import sys

def setup_wandb():
    """Setup Weights & Biases for the project."""
    print("🔮 Setting up Weights & Biases...")
    
    # Check if wandb is installed
    try:
        import wandb
        print("✅ wandb is already installed")
    except ImportError:
        print("📦 Installing wandb...")
        os.system(f"{sys.executable} -m pip install wandb")
        import wandb
    
    # Login to wandb
    print("🔑 Please login to Weights & Biases...")
    print("You can either:")
    print("1. Run 'wandb login' in terminal")
    print("2. Set WANDB_API_KEY environment variable")
    print("3. Login interactively below")
    
    try:
        wandb.login()
        print("✅ Successfully logged in to Weights & Biases")
    except Exception as e:
        print(f"⚠️ Login failed: {e}")
        print("You can continue without login, but logging will be disabled")
    
    print("\n🎯 Weights & Biases setup complete!")
    print("Your training runs will be logged to: https://wandb.ai/")

if __name__ == "__main__":
    setup_wandb()
