#!/bin/bash
# CLAP unfused + SAM + Focal Loss training
# Expected: Score 65-67 (improved from 64.14 baseline)
#
# Usage: bash scripts/train_CLAP_SAM.sh

set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="1 2 3 4 5"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

echo "============================================"
echo " CLAP unfused + SAM + Focal Training"
echo " Seeds: $SEEDS"
echo "============================================"

for s in $SEEDS; do
    TAG="CLAP_SAM_${s}"
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
        --learning_rate 2e-5 \
        --weight_decay 1e-6 \
        --cosine \
        --sample_rate 48000 \
        --model laion/clap-htsat-unfused \
        --model_type ClapModel \
        --meta_mode all \
        --test_fold official \
        --pad_types repeat \
        --ma_update --ma_beta 0.5 \
        --method ce \
        --num_workers 4 \
        --print_freq 50 \
        --focal_loss --focal_gamma 2.0 --focal_alpha auto \
        --sam --sam_rho 0.03 \
        --label_smoothing 0.05

    echo ">>> Finished seed $s at $(date)"
done

echo "============================================"
echo " All CLAP+SAM trainings done!"
echo "============================================"
