# Vitis AI FPGA Deployment Workflow

Tài liệu này mô tả luồng triển khai repo-relative cho pipeline PyTorch/Vitis AI. Các đường dẫn mặc định không phụ thuộc vào user home và output sinh ra nằm dưới `artifacts/`.

## 1. Cấu trúc liên quan

```text
Parallel_Computing_on_FPGA/
├── data/
│   ├── samples/ICBHI_final_database/
│   ├── samples/labels.txt
│   └── combined/
│       ├── audio/
│       └── labels.csv
├── python/
│   ├── common/paths.py
│   ├── training/
│   │   └── distillation_02.py
│   └── quantization/
│       ├── generate_calib_data.py
│       ├── quantize_distillation_03.py
│       └── legacy/
├── fpga/vitis_ai_flow/
│   ├── arch.json
│   └── quantize.py
└── artifacts/
    ├── training/distillation_v2/
    ├── quantization/calibration_data/
    ├── quantization/vitis_qat_v3/
    ├── quantization/vitis_ai_flow/
    └── compile/
```

Legacy export/PTQ scripts such as `validate_and_export_dpu.py`, `create_calib_dataset.py`, and older `quantize_distillation_*` variants are preserved in `python/quantization/legacy/` for reproduction only.

## 2. Active deployment path

The active path is:

1. Train or distill a PyTorch student model.
2. Generate calibration samples.
3. Run the active Vitis QAT/quantization script.
4. Compile the exported `.xmodel` with `vai_c_xir` for Ultra96-V2/DPU.

## 3. Prepare training artifacts

From repository root:

```bash
make distill
```

Equivalent direct command:

```bash
python3 python/training/distillation_02.py \
  --artifact_root artifacts/training
```

Default distillation outputs are written to:

```text
artifacts/training/distillation_v2/
```

The quantization script defaults to the fold-0 student checkpoint:

```text
artifacts/training/distillation_v2/checkpoints/student_fold_0_best.pt
```

You can override this with `--student_ckpt`.

## 4. Generate calibration data

```bash
make calib-data
```

Equivalent direct command:

```bash
python3 python/quantization/generate_calib_data.py \
  --artifact_root artifacts
```

Default output:

```text
artifacts/quantization/calibration_data/
```

## 5. Run active Vitis QAT quantization

Run inside a Vitis AI PyTorch environment when `pytorch_nndct` is required. A typical Docker mount keeps the repo at `/workspace`:

```bash
docker run --gpus all -it --rm \
  -v "$PWD":/workspace \
  -w /workspace \
  xilinx/vitis-ai-pytorch-gpu:3.5.0.001
```

Then run:

```bash
make quantize-vitis
```

Equivalent direct command:

```bash
python3 python/quantization/quantize_distillation_03.py \
  --artifact_root artifacts \
  --student_ckpt artifacts/training/distillation_v2/checkpoints/student_fold_0_best.pt \
  --calib_dir artifacts/quantization/calibration_data \
  --output_dir artifacts/quantization/vitis_qat_v3
```

Default output:

```text
artifacts/quantization/vitis_qat_v3/
```

## 6. Optional fpga/vitis_ai_flow helper

The helper at `fpga/vitis_ai_flow/quantize.py` remains available for the FPGA-specific Vitis flow:

```bash
make quantize-flow
```

Equivalent direct command:

```bash
python3 fpga/vitis_ai_flow/quantize.py \
  --artifact_root artifacts \
  --model_dir artifacts/training/copd_binary \
  --calib_dir artifacts/quantization/calibration_data/cwt_images \
  --output_dir artifacts/quantization/vitis_ai_flow
```

Default output:

```text
artifacts/quantization/vitis_ai_flow/
```

## 7. Compile `.xmodel` for Ultra96-V2

Run `vai_c_xir` inside the Vitis AI compiler environment. Adjust the `-x` file name to the `.xmodel` produced by the quantization step.

```bash
mkdir -p artifacts/compile/ultra96

vai_c_xir \
  -x artifacts/quantization/vitis_qat_v3/COPDClassifier_int.xmodel \
  -a fpga/vitis_ai_flow/arch.json \
  -o artifacts/compile/ultra96 \
  -n copd_classifier_u96
```

Compiled deployment artifacts should stay under:

```text
artifacts/compile/ultra96/
```

## 8. Verification checklist

```bash
make py-compile
make help
```

If CMake is installed:

```bash
make build
make test-cpp
```

Before committing, confirm generated outputs are ignored and `.gitkeep` skeleton files remain trackable:

```bash
git check-ignore -v artifacts/training/model.pth
git check-ignore -v artifacts/training/.gitkeep
```
