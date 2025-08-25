#!/usr/bin/env python3
"""
Script for testing different dataset sizes for Amazon Fashion.
Helps choose optimal size for your goals.
"""

import time
import subprocess
import sys
from pathlib import Path

def test_dataset_size(size, output_dir):
    """Tests dataset creation for given size."""
    print(f"\n{'='*60}")
    print(f"TESTING DATASET SIZE: {size:,} products")
    print(f"{'='*60}")
    
    # Clean previous directory
    if Path(output_dir).exists():
        import shutil
        shutil.rmtree(output_dir)
    
    # Start timing
    start_time = time.time()
    
    try:
        # Run dataset preparation
        cmd = [
            sys.executable, "prepare_fashion_multitask.py",
            "--max_items", str(size),
            "--output_dir", output_dir
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print(f"Starting processing...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"SUCCESS! Dataset created in {duration:.1f} seconds")
            
            # Analyze results
            analyze_results(output_dir, size, duration)
            
        else:
            print(f"ERROR creating dataset:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: Processing {size:,} products took more than an hour")
    except Exception as e:
        print(f"ERROR: {e}")

def analyze_results(output_dir, size, duration):
    """Analyzes dataset creation results."""
    try:
        # Count dialogues
        train_file = Path(output_dir) / "conversations" / "train.jsonl"
        val_file = Path(output_dir) / "conversations" / "validation.jsonl"
        
        if train_file.exists() and val_file.exists():
            with open(train_file, 'r', encoding='utf-8') as f:
                train_count = sum(1 for _ in f)
            
            with open(val_file, 'r', encoding='utf-8') as f:
                val_count = sum(1 for _ in f)
            
            total_dialogs = train_count + val_count
            expected_dialogs = size * 5  # 5 task types per product
            
            print(f"\nRESULTS:")
            print(f"   Products: {size:,}")
            print(f"   Expected dialogues: {expected_dialogs:,}")
            print(f"   Created dialogues: {total_dialogs:,}")
            print(f"   Train: {train_count:,}")
            print(f"   Validation: {val_count:,}")
            print(f"   Time: {duration:.1f} sec")
            print(f"   Speed: {size/duration:.1f} products/sec")
            
            # Check task types
            check_task_types(train_file)
            
        else:
            print(f"Dataset files not found in {output_dir}")
            
    except Exception as e:
        print(f"Error analyzing results: {e}")

def check_task_types(train_file):
    """Checks task type distribution."""
    try:
        import json
        
        task_counts = {}
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    task_type = data.get('task_type', 'unknown')
                    task_counts[task_type] = task_counts.get(task_type, 0) + 1
                except json.JSONDecodeError:
                    continue
        
        print(f"\nTASK TYPE DISTRIBUTION:")
        for task_type, count in sorted(task_counts.items()):
            print(f"   {task_type}: {count:,}")
            
    except Exception as e:
        print(f"Error checking task types: {e}")

def main():
    """Main function."""
    print("TESTING AMAZON FASHION DATASET SIZES")
    print("="*60)
    
    # Test different sizes
    test_sizes = [
        (100, "fashion_test_100"),
        (500, "fashion_test_500"),
        (1000, "fashion_test_1000"),
        (5000, "fashion_test_5000"),
        (10000, "fashion_test_10000")
    ]
    
    print(f"Will test {len(test_sizes)} dataset sizes")
    print(f"This will take approximately 1-2 hours")
    
    # Ask user
    response = input("\nContinue? (y/n): ").lower().strip()
    if response not in ['y', 'yes', 'да', 'д']:
        print("Testing cancelled")
        return
    
    # Run tests
    for size, output_dir in test_sizes:
        test_dataset_size(size, output_dir)
        
        # Pause between tests
        if size != test_sizes[-1][0]:  # Not last test
            print(f"\nPausing 10 seconds before next test...")
            time.sleep(10)
    
    print(f"\n{'='*60}")
    print(f"TESTING COMPLETED!")
    print(f"{'='*60}")
    print(f"Results saved in directories:")
    for size, output_dir in test_sizes:
        print(f"  {size:,} products -> {output_dir}")
    
    print(f"\nRECOMMENDATIONS:")
    print(f"  - For debugging: use 100-500 products")
    print(f"  - For development: use 1K-5K products")
    print(f"  - For training: use 10K+ products")
    print(f"  - For production: use full dataset (2.2M products)")

if __name__ == "__main__":
    main()

