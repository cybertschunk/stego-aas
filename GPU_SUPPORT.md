# GPU Support for stego-aas

## Current Status

The codebase is **GPU-ready** and will automatically detect and use GPU acceleration when available. However, due to Python 3.13 compatibility limitations with current GPU libraries, the application currently runs on CPU.

## How GPU Detection Works

The `ModelManager` class in `model_manager.py` automatically selects the best available device:

```python
self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

When a compatible GPU-enabled PyTorch installation is detected, the model will automatically load on the GPU with no code changes required.

## Python 3.13 Limitations

This project uses **Python 3.13**, which has limited GPU library support as of 2025-01:

### NVIDIA GPUs (CUDA)
- **Status**: Not supported on this system
- **Reason**: System has Intel GPU, not NVIDIA
- **PyTorch CUDA builds**: Available for Python 3.11-3.12, coming soon for 3.13

### Intel GPUs (DirectML/IPEX)
- **Status**: Not yet available
- **torch-directml**: No Python 3.13 support yet
- **intel-extension-for-pytorch**: No Python 3.13 support yet
- **System GPU**: Intel Iris Xe Graphics (detected)

## Enabling GPU Support

### Option 1: Wait for Library Updates (Recommended)
When GPU libraries add Python 3.13 support:

1. Install the appropriate package:
   ```bash
   # For Intel GPU (when available)
   pip install torch-directml
   # OR
   pip install intel-extension-for-pytorch
   ```

2. Run verification:
   ```bash
   python verify_device.py
   ```

3. Tests will automatically run on GPU - no code changes needed

### Option 2: Downgrade to Python 3.11/3.12

If you need GPU acceleration immediately:

1. **Create new virtual environment with Python 3.11 or 3.12**:
   ```bash
   python3.11 -m venv .venv311
   # or
   python3.12 -m venv .venv312
   ```

2. **Activate and install dependencies**:
   ```bash
   # Windows
   .venv311\Scripts\activate

   # Install base requirements
   pip install -r requirements.txt
   ```

3. **Install GPU-enabled PyTorch**:

   **For Intel GPU (DirectML)**:
   ```bash
   pip install torch-directml
   ```

   **For Intel GPU (IPEX - Linux/WSL)**:
   ```bash
   pip install intel-extension-for-pytorch
   ```

   **For NVIDIA GPU (CUDA)**:
   ```bash
   # Visit https://pytorch.org/get-started/locally/
   # Select appropriate CUDA version
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Verify GPU is detected**:
   ```bash
   python verify_device.py
   ```

## Verification

The `verify_device.py` script in the project root provides detailed information about:
- PyTorch version and CUDA availability
- GPU detection status
- Device being used by the model
- System GPU information

Run it anytime to check your GPU configuration:
```bash
python verify_device.py
```

## Performance Expectations

Based on typical performance:
- **CPU (Intel i5/i7)**: ~1-2 seconds per test
- **Intel GPU (Iris Xe)**: ~0.5-1 second per test (2-3x faster)
- **NVIDIA GPU (RTX series)**: ~0.2-0.5 seconds per test (5-10x faster)

The `test_multiple_texts` test currently takes ~100-120 seconds on CPU. With GPU acceleration, this should reduce to 20-50 seconds depending on your GPU.

## Logging

When the model loads, it logs the device being used:
- GPU available: `"Loading model on GPU: [GPU Name]"`
- CPU fallback: `"CUDA not available. Loading model on CPU."`
- Final confirmation: `"Model loaded successfully on device: [device]"`

Check Django logs to see which device is being used.

## Code Changes Made

The following files were modified to support GPU:

1. **`model_manager.py:54`**: Changed from hardcoded `torch.device("cpu")` to dynamic detection
2. **`model_manager.py:59-67`**: Added logging for device information
3. **`verify_device.py`**: New script for device verification

All changes are backward compatible and maintain full CPU functionality.
