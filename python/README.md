# Python ML Pipeline

Python code is split into active pipeline scripts and legacy experiments. Active scripts use `python/common/paths.py` for repo-relative defaults and write generated outputs to `artifacts/`.

## Structure

```text
python/
├── common/
│   └── paths.py                         # Repo/data/artifact path constants
├── preprocessing/
│   ├── combine_dataset.py               # Build data/combined from source datasets
│   ├── preprocessing.py                 # Resample/filter/segment combined audio
│   ├── preprocess_audio_dsp.py          # DSP/CWT image preprocessing helper
│   └── wavelet_transform.py             # CWT spectrogram generation
├── training/
│   ├── train_mobilenetv2.py             # Active MobileNetV2 3-class training
│   ├── train_efficientnet_b0.py         # Active EfficientNet-B0 baseline
│   ├── distillation_02.py               # Active teacher-student distillation
│   └── experiments/legacy/              # Older CNN/MobileNet experiments
├── quantization/
│   ├── generate_calib_data.py           # Active calibration data generator
│   ├── quantize_distillation_03.py      # Active Vitis QAT path
│   └── legacy/                          # Older PTQ/export/validation variants
├── Inspector/                           # Vitis AI inspection utilities
├── layer1_3_experiments/                # Layer 1-3 feature/RF experiments
└── visualizations/                      # Paper figures and analysis plots
```

## Default data and output paths

Default inputs:

```text
data/samples/ICBHI_final_database/
data/samples/labels.txt
data/samples_02/
data/combined/audio/
data/combined/labels.csv
```

Default generated outputs:

```text
artifacts/training/mobilenetv2_3class_raw/
artifacts/training/efficientnet_b0_3class/
artifacts/training/distillation_v2/
artifacts/training/layer1_3/
artifacts/quantization/calibration_data/
artifacts/quantization/vitis_qat_v3/
artifacts/paper_figures/
```

Most active scripts accept `--data_dir`, `--output_dir`, or `--artifact_root` so runs can be redirected without editing source code.

## Recommended run order

From repository root:

```bash
python3 python/preprocessing/combine_dataset.py --output-dir data/combined
python3 python/preprocessing/preprocessing.py \
  --data_dir data/combined/audio \
  --labels_csv data/combined/labels.csv \
  --output_dir data/combined/processed_audio
python3 python/preprocessing/wavelet_transform.py \
  --data_dir data/combined/processed_audio \
  --output_dir data/combined/spectrograms

python3 python/training/train_mobilenetv2.py \
  --artifact_root artifacts/training
python3 python/training/train_efficientnet_b0.py \
  --artifact_root artifacts/training
python3 python/training/distillation_02.py \
  --artifact_root artifacts/training

python3 python/quantization/generate_calib_data.py \
  --artifact_root artifacts
python3 python/quantization/quantize_distillation_03.py \
  --artifact_root artifacts
```

Equivalent Make targets are available from the repository root:

```bash
make preprocess-combined
make preprocess-audio
make wavelet
make train-mobilenet
make train-efficientnet
make distill
make calib-data
make quantize-vitis
```

## Legacy scripts

Legacy scripts are preserved but excluded from the main path:

- Training legacy: `python/training/experiments/legacy/`
- Quantization/export legacy: `python/quantization/legacy/`

Use them only when reproducing older experiments or comparing alternate approaches.
