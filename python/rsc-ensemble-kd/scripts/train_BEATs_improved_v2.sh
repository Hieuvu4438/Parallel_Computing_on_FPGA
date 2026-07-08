#!/bin/bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export LD_LIBRARY_PATH=/home/haipd/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd /home/haipd/Parallel_Computing_on_FPGA/python/rsc-ensemble-kd

echo "=== BEATs improved v2: Focal + Mixup + Label Smoothing ==="
taskset -c 0-7 python -u main.py --tag BEATs_improved_v2 \
    --dataset icbhi --seed 1 --data_folder ./data/ \
    --soft_label_mode none --class_split lungsound --n_cls 4 \
    --epochs 50 --batch_size 8 --optimizer adam --learning_rate 5e-6 \
    --weight_decay 1e-4 --cosine --warm --sample_rate 16000 --model beats \
    --test_fold official --pad_types repeat \
    --method ce --num_workers 0 --print_freq 50 \
    --focal_loss --focal_gamma 2.0 --focal_alpha auto \
    --mixup 0.3 \
    --label_smoothing 0.1

echo "=== DONE ==="
