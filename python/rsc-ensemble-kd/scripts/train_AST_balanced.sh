#!/bin/bash
# AST AudioSet-pretrained + aggressive class balancing
# Expected: Score 62-65 (improved from 59.97 baseline)
#
# Key: AST has high S_p (85.88) but low S_e (34.07)
# Strategy: focal gamma=3.0 + class-balanced sampling + mixup + SAM
#
# Usage: bash scripts/train_AST_balanced.sh

set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="1 2 3 4 5"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

echo "============================================"
echo " AST + Aggressive Class Balancing"
echo " Seeds: $SEEDS"
echo "============================================"

for s in $SEEDS; do
    TAG="AST_balanced_${s}"
    echo ">>> Training seed $s at $(date)"

    CUDA_VISIBLE_DEVICES=$GPU python main.py \
        --tag "$TAG" \
        --dataset icbhi \
        --seed "$s" \
        --data_folder ./data/ \
        --soft_label_mode "none" \
        --class_split lungsound \
        --n_cls 4 \
        --epochs 60 \
        --batch_size 16 \
        --optimizer adam \
        --learning_rate 1e-5 \
        --weight_decay 1e-4 \
        --cosine \
        --warm --warm_epochs 5 \
        --sample_rate 16000 \
        --model ast \
        --from_sl_official \
        --audioset_pretrained \
        --test_fold official \
        --num_workers 4 \
        --print_freq 50 \
        --specaug_policy icbhi_ast_sup \
        --focal_loss --focal_gamma 3.0 --focal_alpha auto \
        --class_balanced_sampling \
        --mixup 0.3 \
        --label_smoothing 0.15 \
        --sam --sam_rho 0.05

    echo ">>> Finished seed $s at $(date)"
done

echo "============================================"
echo " All AST balanced trainings done!"
echo "============================================"
