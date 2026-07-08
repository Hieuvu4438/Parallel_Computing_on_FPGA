#!/bin/bash
# Train BEATs ensemble with 5 different seeds
# CPU-limited: taskset + OMP_NUM_THREADS + num_workers=0

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export LD_LIBRARY_PATH=/home/haipd/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd /home/haipd/Parallel_Computing_on_FPGA/python/rsc-ensemble-kd

SEEDS="1 2 3 4 5"

for s in $SEEDS
do
    TAG="BEATs_ensemble_${s}"
    echo "=== Training BEATs seed=$s ==="
    taskset -c 0-7 python -u main.py --tag $TAG \
        --dataset icbhi --seed $s --data_folder ./data/ \
        --soft_label_mode none --class_split lungsound --n_cls 4 \
        --epochs 50 --batch_size 8 --optimizer adam --learning_rate 5e-6 \
        --weight_decay 1e-4 --cosine --warm --sample_rate 16000 --model beats \
        --test_fold official --pad_types repeat \
        --method ce --num_workers 0 --print_freq 50

    echo "=== BEATs seed=$s done ==="
done

echo "=== ALL BEATs ENSEMBLE DONE ==="
