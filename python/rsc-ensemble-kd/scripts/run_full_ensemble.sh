#!/bin/bash
# Full Multi-Model Ensemble Pipeline
#
# Step 1: Train larger_clap_general (5 seeds)
# Step 2: Train clap-htsat-fused (5 seeds)
# Step 3: Evaluate multi-model ensemble with all available logits
#
# Each training step is guarded by CPU monitor (max 200% CPU).
# Estimated time: ~4-6 hours total (depends on GPU speed)
#
# Usage: bash scripts/run_full_ensemble.sh
# Or with specific GPU: CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_ensemble.sh

set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${CUDA_VISIBLE_DEVICES:-0}"
MAX_CPU=200
LOGDIR="logs/ensemble_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo " Multi-Model Ensemble Pipeline"
echo " GPU: $GPU"
echo " CPU limit: ${MAX_CPU}%"
echo " Log dir: $LOGDIR"
echo " Start: $(date)"
echo "============================================"

# ── Pre-flight checks ──
echo ""
echo "[Pre-flight] Checking GPU..."
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

echo "[Pre-flight] Checking CPU..."
CURRENT_CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}' | cut -d'.' -f1)
echo "  Current CPU: ${CURRENT_CPU}%"
if [ "$CURRENT_CPU" -gt "$MAX_CPU" ]; then
    echo "  ERROR: CPU too high! Aborting."
    exit 1
fi
echo "  OK"

# ── Step 1: Train larger_clap_general ──
echo ""
echo "============================================"
echo " Step 1/3: Training larger_clap_general"
echo "============================================"

SEEDS="1 2 3 4 5"
for s in $SEEDS; do
    TAG="BTS_larger_clap_${s}"
    echo ">>> Training seed $s (tag: $TAG) at $(date)"

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
        --model "laion/larger_clap_general" \
        --model_type ClapModel \
        --meta_mode all \
        --test_fold official \
        --pad_types repeat \
        --ma_update \
        --ma_beta 0.5 \
        --method ce \
        --num_workers 4 \
        --print_freq 50 \
        2>&1 | tee "$LOGDIR/larger_clap_seed${s}.log"

    # CPU check after each training
    CPU_NOW=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}' | cut -d'.' -f1)
    echo "  CPU after training: ${CPU_NOW}%"
done

# ── Step 2: Train clap-htsat-fused ──
echo ""
echo "============================================"
echo " Step 2/3: Training clap-htsat-fused"
echo "============================================"

for s in $SEEDS; do
    TAG="BTS_fused_${s}"
    echo ">>> Training seed $s (tag: $TAG) at $(date)"

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
        --model "laion/clap-htsat-fused" \
        --model_type ClapModel \
        --meta_mode all \
        --test_fold official \
        --pad_types repeat \
        --ma_update \
        --ma_beta 0.5 \
        --method ce \
        --num_workers 4 \
        --print_freq 50 \
        2>&1 | tee "$LOGDIR/fused_seed${s}.log"

    CPU_NOW=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}' | cut -d'.' -f1)
    echo "  CPU after training: ${CPU_NOW}%"
done

# ── Step 3: Extract logits and run ensemble ──
echo ""
echo "============================================"
echo " Step 3/3: Extract logits & Ensemble"
echo "============================================"

# Extract logits from larger_clap models
mkdir -p teacher_logits_larger_clap
for s in $SEEDS; do
    CKPT="save/icbhi_laion_larger_clap_general_ce_all_BTS_larger_clap_${s}/best.pth"
    if [ -f "$CKPT" ]; then
        echo "Extracting logits from $CKPT"
        python scripts/extract_logits.py \
            --checkpoint "$CKPT" \
            --model "laion/larger_clap_general" \
            --model_type ClapModel \
            --output_dir "teacher_logits_larger_clap/seed_${s}" \
            --num_workers 4
    fi
done

# Extract logits from fused models
mkdir -p teacher_logits_fused
for s in $SEEDS; do
    CKPT="save/icbhi_laion_clap-htsat-fused_ce_all_BTS_fused_${s}/best.pth"
    if [ -f "$CKPT" ]; then
        echo "Extracting logits from $CKPT"
        python scripts/extract_logits.py \
            --checkpoint "$CKPT" \
            --model "laion/clap-htsat-fused" \
            --model_type ClapModel \
            --output_dir "teacher_logits_fused/seed_${s}" \
            --num_workers 4
    fi
done

# Run ensemble evaluation
echo ""
echo "Running multi-model ensemble..."
python scripts/ensemble_multi_model.py \
    --logits_dirs teacher_logits teacher_logits_larger_clap teacher_logits_fused \
    --logits_dirs_beats teacher_logits_beats \
    --n_cls 4 \
    --grid_steps 20 \
    2>&1 | tee "$LOGDIR/ensemble_results.log"

echo ""
echo "============================================"
echo " Pipeline Complete!"
echo " Results: save/ensemble_results.json"
echo " Logs: $LOGDIR/"
echo " End: $(date)"
echo "============================================"
