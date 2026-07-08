#!/bin/bash
# Ablation: Focal Loss only (highest impact expected)
# Tests focal loss with auto class weights against the baseline.
# CPU-limited: taskset + OMP_NUM_THREADS + num_workers=0

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export LD_LIBRARY_PATH=/home/haipd/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd /home/haipd/Parallel_Computing_on_FPGA/python/rsc-ensemble-kd

MODEL="laion/clap-htsat-unfused"
SEED="1"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="BTS_focal_${s}"
        taskset -c 0-7 python -u main.py --tag $TAG \
                                        --dataset icbhi \
                                        --seed $s \
                                        --data_folder ./data/ \
                                        --soft_label_mode "mean_5" \
                                        --class_split lungsound \
                                        --n_cls 4 \
                                        --epochs 50 \
                                        --batch_size 8 \
                                        --optimizer adam \
                                        --learning_rate 5e-5 \
                                        --weight_decay 1e-6 \
                                        --cosine \
                                        --sample_rate 48000 \
                                        --model $m \
                                        --model_type ClapModel \
                                        --meta_mode all \
                                        --test_fold official \
                                        --pad_types repeat \
                                        --ma_update \
                                        --ma_beta 0.5 \
                                        --method ce \
                                        --num_workers 0 \
                                        --print_freq 100 \
                                        --focal_loss \
                                        --focal_gamma 2.0 \
                                        --focal_alpha auto
    done
done
