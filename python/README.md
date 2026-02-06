# CNN Training for Layer 4 - Respiratory Sound Classification

## Overview

This directory contains Python scripts for training and exporting CNN models for Layer 4 of the Cascaded Framework for Respiratory Sound Analysis.

## Files

| File | Description |
|------|-------------|
| `train_cnn_layer4.py` | **5-class** training based on patient diagnosis (Healthy, URTI, LRTI, Bronchiectasis, COPD) |
| `train_cnn_4class.py` | **4-class** training based on cycle annotations (Normal, Crackle, Wheeze, Both) - **Recommended for C++ integration** |
| `requirements.txt` | Python dependencies |

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install PyTorch (visit https://pytorch.org for CUDA version)
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

### 2. Train Model (4-class for C++ Integration)

```bash
python train_cnn_4class.py \
    --data_path "D:/PROJECTS/Parallel_Computing_on_FPGA/data/ICBHI_final_database" \
    --output_dir "./output_4class" \
    --epochs 50 \
    --batch_size 32
```

### 3. Output Files

After training, you will find in `output_dir`:

| File | Description |
|------|-------------|
| `best_model.pth` | PyTorch model weights |
| `respiratory_4class.onnx` | ONNX model for C++ inference |
| `training_results.png` | Training curves & confusion matrix |

## Training Strategy

### Phase 1: Classifier Training (15 epochs)
- Backbone (MobileNetV2) is **frozen**
- Only the classifier head is trained
- Learning rate: `1e-3`

### Phase 2: Fine-tuning (35 epochs)
- All layers are **unfrozen**
- Learning rate: `1e-5` (very low to prevent catastrophic forgetting)
- Cosine annealing scheduler

## Key Features

### 1. Wavelet Transform (CWT with Morlet)
```python
# Continuous Wavelet Transform for time-frequency representation
# Better than STFT for non-stationary respiratory sounds
wavelet = WaveletTransform(
    wavelet='morl',
    num_scales=128,
    freq_range=(50, 2000),
    output_size=224
)
```

### 2. Subject-Independent Split
```python
# Ensures no patient appears in both train and test sets
# Critical for fair evaluation in medical applications
train_idx, val_idx, test_idx = subject_split(cycles, labels, patient_ids)
```

### 3. Class Imbalance Handling
```python
# WeightedRandomSampler for balanced training
# Addresses ICBHI's imbalanced class distribution
sampler = WeightedRandomSampler(weights, len(weights))
```

### 4. Data Augmentation
- Random gain adjustment (0.8x - 1.2x)
- Additive Gaussian noise
- Time shift

## Integration with C++ Code

After training, copy the ONNX file to the project:

```bash
cp output_4class/respiratory_4class.onnx ../models/
```

Then in C++, enable ONNX Runtime:

```cpp
// In CnnInference.cpp, uncomment:
#define USE_ONNX_RUNTIME

// Load model:
CnnInference cnn;
cnn.loadModel("../models/respiratory_4class.onnx");
```

## Expected Results

Based on ICBHI 2017 dataset:

| Metric | Target | Notes |
|--------|--------|-------|
| Accuracy | 70-80% | Subject-independent |
| Macro F1 | 60-70% | Due to class imbalance |
| Sensitivity (Crackle) | 70%+ | Important for diagnosis |
| Specificity (Normal) | 80%+ | Avoid false positives |

> **Note**: Achieving 98% accuracy is challenging with ICBHI due to:
> - Severe class imbalance
> - Subject-independent evaluation
> - Noisy recordings from various equipment

## Command Line Arguments

```
--data_path       Path to ICBHI_final_database
--output_dir      Directory for output files
--epochs          Total training epochs (default: 50)
--phase1_epochs   Epochs for frozen backbone (default: 15)
--batch_size      Batch size (default: 32)
--dropout         Dropout rate (default: 0.3)
--num_workers     DataLoader workers (default: 4)
```

## Hardware Requirements

- **GPU**: Recommended (NVIDIA with CUDA)
- **RAM**: 8GB+ (16GB recommended)
- **Training time**: ~2-3 hours with GPU, 10+ hours with CPU

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python train_cnn_4class.py --batch_size 16
```

### Slow Training
```bash
# Use GPU
export CUDA_VISIBLE_DEVICES=0

# Or reduce workers on Windows
python train_cnn_4class.py --num_workers 0
```

### Missing Audio Files
Ensure ICBHI dataset is complete with both `.wav` and `.txt` files.

## References

- ICBHI 2017 Scientific Challenge: https://bhichallenge.med.auth.gr/
- MobileNetV2: https://arxiv.org/abs/1801.04381
- Wavelet Transform: https://pywavelets.readthedocs.io/
