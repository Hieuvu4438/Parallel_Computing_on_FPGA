#!/bin/bash
# BEATs with all improvements: Focal Loss + Mixup + Label Smoothing + SAM
# CPU-limited: taskset + OMP_NUM_THREADS + num_workers=0

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export LD_LIBRARY_PATH=/home/haipd/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd /home/haipd/Parallel_Computing_on_FPGA/python/rsc-ensemble-kd

echo "=== Training BEATs with all improvements ==="
echo "  Focal Loss (gamma=2.0, alpha=auto)"
echo "  Mixup (alpha=0.3)"
echo "  Label Smoothing (epsilon=0.1)"
echo "  SAM (rho=0.05)"
echo ""

taskset -c 0-7 python -u main.py --tag BEATs_improved_1 \
    --dataset icbhi --seed 1 --data_folder ./data/ \
    --soft_label_mode none --class_split lungsound --n_cls 4 \
    --epochs 50 --batch_size 8 --optimizer adam --learning_rate 5e-6 \
    --weight_decay 1e-4 --cosine --warm --sample_rate 16000 --model beats \
    --test_fold official --pad_types repeat \
    --method ce --num_workers 0 --print_freq 50 \
    --focal_loss --focal_gamma 2.0 --focal_alpha auto \
    --mixup 0.3 \
    --label_smoothing 0.1 \
    --sam --sam_rho 0.05

echo "=== BEATs improved training done ==="
