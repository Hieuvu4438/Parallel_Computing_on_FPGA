#!/bin/bash
# Train BTS with laion/clap-htsat-fused (fused cross-modal attention)
# Uses main_custom.py — does NOT modify main.py or any existing code.
#
# Fused version has cross-attention layers between text and audio,
# potentially giving better text-audio alignment for metadata-aware classification.
#
# Usage: bash scripts/train_BTS_fused.sh
# With CPU guard: bash scripts/cpu_guard.sh 200 bash scripts/train_BTS_fused.sh

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="laion/clap-htsat-fused"
SEEDS="1 2 3 4 5"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

echo "============================================"
echo " Training BTS with clap-htsat-fused"
echo " Seeds: $SEEDS"
echo " GPU: $GPU"
echo "============================================"

for s in $SEEDS; do
    TAG="BTS_fused_${s}"
    echo ""
    echo ">>> Training seed $s with tag $TAG"
    echo ">>> Start time: $(date)"

    CUDA_VISIBLE_DEVICES=$GPU python main_custom.py \
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
        --learning_rate 5e-5 \
        --weight_decay 1e-6 \
        --cosine \
        --sample_rate 48000 \
        --model "$MODEL" \
        --model_type ClapModel \
        --meta_mode all \
        --test_fold official \
        --pad_types repeat \
        --ma_update \
        --ma_beta 0.5 \
        --method ce \
        --num_workers 4 \
        --print_freq 50

    echo ">>> Finished seed $s at $(date)"
done

echo ""
echo "============================================"
echo " All fused CLAP trainings done!"
echo "============================================"
