#!/bin/bash
# CACD (Class-Aware Curriculum Distillation) with CNN14 student
# FPGA-deployable respiratory sound classification
#
# Method:
#   Stage 1 (epochs 1-10): Binary KD (normal vs abnormal)
#   Stage 2 (epochs 11-50): Class-aware temperature KD + feature alignment
#
# Student: CNN14 (lightweight, FPGA-friendly)
# Teacher: CLAP ensemble (pre-computed logits + features)
#
# CPU-limited execution

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export LD_LIBRARY_PATH=/home/haipd/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd /home/haipd/Parallel_Computing_on_FPGA/python/rsc-ensemble-kd

SEED="1"

for s in $SEED
do
    TAG="CACD_CNN14_${s}"
    taskset -c 0-7 python -u main.py --tag $TAG \
                                        --dataset icbhi \
                                        --seed $s \
                                        --data_folder ./data/ \
                                        --soft_label_mode "mean_5" \
                                        --class_split lungsound \
                                        --n_cls 4 \
                                        --epochs 50 \
                                        --batch_size 32 \
                                        --optimizer adam \
                                        --learning_rate 1e-4 \
                                        --weight_decay 1e-4 \
                                        --cosine \
                                        --sample_rate 16000 \
                                        --model cnn14 \
                                        --test_fold official \
                                        --pad_types repeat \
                                        --ma_update \
                                        --ma_beta 0.99 \
                                        --method ce \
                                        --num_workers 0 \
                                        --print_freq 50 \
                                        --cacd \
                                        --cacd_T 4.0 \
                                        --cacd_beta 0.5 \
                                        --cacd_alpha 0.5 \
                                        --cacd_feat_weight 0.1 \
                                        --cacd_stage1_epochs 10 \
                                        --teacher_features_dir ./teacher_features

done
