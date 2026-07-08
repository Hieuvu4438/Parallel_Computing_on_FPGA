#!/bin/bash
# Quick verification: 1 epoch, small batch, limited workers
# Runs in ~2-5 minutes to verify everything works before full training

cd /home/haipd/Parallel_Computing_on_FPGA/python/rsc-ensemble-kd

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/home/haipd/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

echo "=== VERIFY SETUP ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader)"
echo "Time: $(date)"
echo "===================="

python -u main.py \
    --tag "verify" \
    --dataset icbhi \
    --seed 1 \
    --data_folder ./data/ \
    --soft_label_mode "mean_5" \
    --class_split lungsound \
    --n_cls 4 \
    --epochs 1 \
    --batch_size 8 \
    --optimizer adam \
    --learning_rate 5e-5 \
    --weight_decay 1e-6 \
    --cosine \
    --sample_rate 48000 \
    --model laion/clap-htsat-unfused \
    --model_type ClapModel \
    --meta_mode all \
    --test_fold official \
    --pad_types repeat \
    --ma_update \
    --ma_beta 0.5 \
    --method ce \
    --num_workers 0 \
    --print_freq 5 \
    --focal_loss \
    --focal_gamma 2.0 \
    --focal_alpha auto

echo "=== VERIFY DONE ==="
