# Respiratory Sound Analysis on FPGA

Triển khai pipeline phân tích âm thanh hô hấp cho nghiên cứu FPGA/Ultra96-V2, gồm C++ cascaded framework, pipeline Python huấn luyện mô hình, distillation, quantization và tài liệu Vitis AI.

## Cấu trúc thư mục

```text
Parallel_Computing_on_FPGA/
├── Makefile                         # Runner trung tâm cho build/test/Python pipeline
├── CMakeLists.txt                   # C++ build system
├── include/                         # C++ public headers
├── src/                             # C++ implementation
├── tests/cpp/                       # C++ tests
├── python/
│   ├── common/paths.py              # Source-of-truth cho repo/data/artifact paths
│   ├── preprocessing/               # Combine dataset, audio preprocessing, CWT
│   ├── training/                    # Pipeline training chính
│   │   └── experiments/legacy/      # Script thử nghiệm cũ
│   ├── quantization/                # Deployment/quantization path chính
│   │   └── legacy/                  # Quantization/export variants cũ
│   ├── layer1_3_experiments/        # Layer 1-3 feature/RF experiments
│   ├── Inspector/                   # Vitis AI inspection helpers
│   └── visualizations/              # Paper figures and analysis plots
├── fpga/vitis_ai_flow/              # Vitis AI arch + quantize helper
├── data/                            # Local datasets and processed data
├── artifacts/                       # Generated outputs, ignored except .gitkeep
├── docs/                            # Workflow docs
└── Paper/                           # LaTeX paper assets/source
```

## Artifact layout

Generated outputs should stay under `artifacts/` instead of being scattered in source folders:

```text
artifacts/
├── training/
│   ├── mobilenetv2_3class_raw/
│   ├── efficientnet_b0_3class/
│   ├── distillation_v2/
│   ├── icbhi_3class_kd/
│   └── layer1_3/
├── quantization/
│   ├── calibration_data/
│   ├── vitis_qat_v3/
│   └── vitis_ai_flow/
├── compile/
├── paper_figures/
└── features/
```

`artifacts/**` is ignored by git, while `.gitkeep` files preserve the directory skeleton. Paper and docs image assets are allowed to be tracked.

## Quick start

```bash
make help
make py-compile
```

C++ build/test, when CMake is installed:

```bash
make build
make test-cpp
make run
```

Python pipeline:

```bash
make preprocess-combined
make preprocess-audio
make wavelet
make train-mobilenet
make train-efficientnet
make distill
make train-kd-icbhi
make calib-data
make quantize-vitis
```

Paper figures:

```bash
make plot-signal
make plot-kd
make plot-teacher-student
make plot-all
```

## Active Python deployment path

The main Python path is:

1. `python/preprocessing/combine_dataset.py`
2. `python/preprocessing/preprocessing.py`
3. `python/preprocessing/wavelet_transform.py`
4. `python/training/train_mobilenetv2.py` or `python/training/train_efficientnet_b0.py`
5. `python/training/distillation_02.py`
6. `python/training/kd_icbhi_3class.py` for the ICBHI-only 3-class teacher-ensemble KD path
7. `python/quantization/generate_calib_data.py`
8. `python/quantization/quantize_distillation_03.py`
9. `fpga/vitis_ai_flow/quantize.py` for the FPGA/Vitis helper flow

Older experiments remain available under `python/training/experiments/legacy/` and `python/quantization/legacy/`, but they are no longer the documented main path.

For the ICBHI-only COPD/Non-COPD/Healthy KD workflow, place raw files in `data/sample_01/ICBHI_final_database/` and subject diagnosis labels in `data/sample_01/labels.txt`, then run `make train-kd-icbhi`. The script maps subject diagnoses to 3 classes and writes checkpoints, soft labels and metrics to `artifacts/training/icbhi_3class_kd/`. See `docs/ICBHI_3CLASS_KD_PIPELINE.md` for the training design, metrics, KD formulas and WandB logging details.

## Notes

The Python/Paper training flow is organized around 3 disease classes: `Healthy`, `COPD`, and `Non-COPD`. The C++ cascaded framework still models the paper-style staged respiratory-sound/event pipeline and should be treated separately from the 3-class disease-label refactor.
