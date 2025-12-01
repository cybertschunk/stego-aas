#!/usr/bin/env python
"""
GPU/CPU Device Verification Script

This script verifies which device (CPU or GPU) is being used for model inference.
It provides detailed information about:
- PyTorch CUDA availability and version
- GPU hardware information (if available)
- Model Manager device configuration
- Actual device where model is loaded

Usage:
    python verify_device.py

Expected output:
- If GPU available: Shows GPU name, memory, and confirms GPU usage
- If GPU not available: Confirms CPU usage (expected for Python 3.13)

This is useful for:
- Troubleshooting GPU configuration issues
- Verifying GPU acceleration is enabled
- Checking PyTorch installation correctness
"""
import sys
import os

# Add the Django project to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stego-aas', 'stegoaas'))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stegoaas.settings')
import django
django.setup()

import torch
from sparsamp_app.model_manager import get_model_manager

def main():
    print("=" * 60)
    print("Device Configuration Verification")
    print("=" * 60)

    # Check PyTorch CUDA availability
    print(f"\nPyTorch CUDA Information:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"  CUDA not available - will use CPU")

    # Load model and check device
    print(f"\nLoading model...")
    model_manager = get_model_manager()
    model_manager.load()

    print(f"\nModel Manager Device:")
    print(f"  Device: {model_manager.device}")
    print(f"  Model is on: {next(model_manager.model.parameters()).device}")

    # Verify with a simple tensor operation
    print(f"\nVerifying tensor operations:")
    test_tensor = torch.randn(1, 10).to(model_manager.device)
    print(f"  Test tensor device: {test_tensor.device}")

    print("\n" + "=" * 60)
    if torch.cuda.is_available():
        print("[OK] GPU is available and will be used for computations")
    else:
        print("[OK] GPU not available - using CPU (expected behavior)")
    print("=" * 60)

if __name__ == "__main__":
    main()
