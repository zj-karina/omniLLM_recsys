#!/usr/bin/env python3
"""
Script to fix missing dependencies for Weights & Biases.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def check_package(package):
    """Check if a package is installed."""
    try:
        __import__(package)
        print(f"✅ {package} is already installed")
        return True
    except ImportError:
        print(f"⚠️ {package} is not installed")
        return False

def main():
    """Main function to fix dependencies."""
    print("🔧 FIXING DEPENDENCIES FOR WEIGHTS & BIASES")
    print("=" * 50)
    
    # Required packages for wandb
    required_packages = [
        "requests",
        "pyyaml", 
        "rich",
        "wandb"
    ]
    
    print("Checking required packages...")
    
    missing_packages = []
    for package in required_packages:
        if not check_package(package):
            missing_packages.append(package)
    
    if not missing_packages:
        print("\n🎉 All required packages are already installed!")
        return
    
    print(f"\n📦 Installing {len(missing_packages)} missing packages...")
    
    success_count = 0
    for package in missing_packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation summary:")
    print(f"   Successfully installed: {success_count}/{len(missing_packages)}")
    
    if success_count == len(missing_packages):
        print("\n🎉 All dependencies fixed!")
        print("\nNext steps:")
        print("1. Run 'wandb login' to authenticate")
        print("2. Run 'make test-logging' to test logging")
        print("3. Run 'make train' to start training with logging")
    else:
        print("\n⚠️ Some packages failed to install.")
        print("You may need to install them manually:")
        for package in missing_packages:
            print(f"   pip install {package}")

if __name__ == "__main__":
    main()







