#!/usr/bin/env python3
"""
Quick test for NaN loss fixes.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def quick_test():
    """Quick test of model initialization."""
    print("🧪 Quick test of model initialization...")
    
    try:
        from any2any_trainer.models.recommendation import RecommendationModel
        
        # Create small model for testing
        model = RecommendationModel(
            ckpt="Qwen/Qwen2.5-Omni-7B",
            id_vocab_size=100,  # Very small vocab
            id_dim=64,
            fusion_dim=128,
            reduced_dim=128,
            device='cpu'
        )
        
        print("✅ Model created successfully")
        
        # Test forward pass
        dummy_text = ["Test product"]
        dummy_ids = [torch.tensor([1, 2])]
        dummy_labels = torch.tensor([3])
        
        with torch.no_grad():
            output = model(dummy_text, dummy_ids, dummy_labels)
        
        print(f"✅ Forward pass successful")
        print(f"   Loss: {output['loss']:.6f}")
        print(f"   Logits mean: {output['logits'].mean():.6f}")
        
        # Check for NaN
        if torch.isnan(output['loss']) or torch.isnan(output['logits']).any():
            print("❌ NaN detected!")
            return False
        else:
            print("✅ No NaN detected!")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🎉 Model should work! Try training now.")
    else:
        print("\n❌ Model has issues. Check the errors above.")
    sys.exit(0 if success else 1)
