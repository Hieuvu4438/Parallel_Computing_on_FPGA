#!/bin/bash
# BEATs + Focal Loss + LDAM training (no SAM — SAM causes NaN with batch_size=8)
# Expected: Score 63-65 (improved from 61.33 baseline)
#
# Key improvements over baseline:
# - Focal Loss with auto class weights for imbalance
# - LDAM for class-aware margins
# - SpecAugment for regularization
# - Higher learning rate with warmup
#
# Usage: bash scripts/train_BEATs_SAM.sh

set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="1 2 3 4 5"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

echo "============================================"
echo " BEATs + Focal + LDAM Training"
echo " Seeds: $SEEDS"
echo "============================================"

for s in $SEEDS; do
    TAG="BEATs_SAM_${s}"
    echo ">>> Training seed $s at $(date)"

    CUDA_VISIBLE_DEVICES=$GPU python main.py \
        --tag "$TAG" \
        --dataset icbhi \
        --seed "$s" \
        --data_folder ./data/ \
        --soft_label_mode "none" \
        --class_split lungsound \
        --n_cls 4 \
        --epochs 50 \
        --batch_size 8 \
        --optimizer adam \
        --learning_rate 3e-5 \
        --weight_decay 1e-4 \
        --cosine \
        --warm --warm_epochs 3 \
        --sample_rate 16000 \
        --model beats \
        --test_fold official \
        --pad_types repeat \
        --num_workers 4 \
        --print_freq 100 \
        --focal_loss --focal_gamma 2.5 --focal_alpha auto \
        --specaug_policy icbhi_ast_sup

    echo ">>> Finished seed $s at $(date)"
done

echo "============================================"
echo " All BEATs+Focal trainings done!"
echo "============================================"
