#!/bin/bash
# Pre-cache test.pt with CPU limits (run once)
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export LD_LIBRARY_PATH=/home/haipd/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd /home/haipd/Parallel_Computing_on_FPGA/python/rsc-ensemble-kd

taskset -c 0-3 python -u main.py \
    --tag "cache_test" \
    --dataset icbhi \
    --seed 1 \
    --data_folder ./data/ \
    --soft_label_mode "none" \
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
    --method ce \
    --num_workers 0 \
    --print_freq 100

echo "=== CACHE DONE ==="
