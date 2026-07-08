#!/bin/bash
# Master SOTA Pipeline: Train All → Ensemble → TTA → Threshold → Stack → Distill
#
# Target: ICBHI Score > 67 on official 60/40 split
#
# Pipeline:
#   Phase 1: Train improved teachers (BEATs+SAM, CLAP+SAM, AST+balanced)
#   Phase 2: Extract logits from all teachers
#   Phase 3: Ensemble with weight optimization + TTA + threshold
#   Phase 4: Stacking meta-learner
#   Phase 5: CACD distillation to CNN14
#
# Usage: bash scripts/run_sota_pipeline.sh
# With CPU guard: bash scripts/cpu_guard.sh 200 bash scripts/run_sota_pipeline.sh

set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${CUDA_VISIBLE_DEVICES:-0}"
MAX_CPU=200
LOGDIR="logs/sota_pipeline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "============================================"
echo " SOTA Pipeline: ICBHI Score > 67"
echo " GPU: $GPU"
echo " Log: $LOGDIR"
echo " Start: $(date)"
echo "============================================"

# ── Pre-flight ──
CURRENT_CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}' | cut -d'.' -f1)
echo "[Pre-flight] CPU: ${CURRENT_CPU}%"
if [ "$CURRENT_CPU" -gt "$MAX_CPU" ]; then
    echo "ERROR: CPU too high! Aborting."
    exit 1
fi

# ═══════════════════════════════════════════════
# Phase 1: Train Improved Teachers
# ═══════════════════════════════════════════════
echo ""
echo "═══ Phase 1: Training Improved Teachers ═══"

# 1a. BEATs + Focal (5 seeds) — no SAM (NaN with batch_size=8)
echo "[1a] BEATs + Focal training..."
for s in 1 2 3 4 5; do
    TAG="BEATs_SAM_${s}"
    echo "  Seed $s at $(date)"
    CUDA_VISIBLE_DEVICES=$GPU python main.py \
        --tag "$TAG" --dataset icbhi --seed "$s" \
        --data_folder ./data/ --soft_label_mode "none" \
        --class_split lungsound --n_cls 4 --epochs 50 --batch_size 8 \
        --optimizer adam --learning_rate 3e-5 --weight_decay 1e-4 \
        --cosine --warm --warm_epochs 3 --sample_rate 16000 \
        --model beats --test_fold official --pad_types repeat \
        --num_workers 4 --print_freq 100 \
        --focal_loss --focal_gamma 2.5 --focal_alpha auto \
        --specaug_policy icbhi_ast_sup \
        2>&1 | tee "$LOGDIR/beats_sam_${s}.log"
done

# 1b. CLAP unfused + SAM + Focal (5 seeds)
echo "[1b] CLAP + SAM training..."
for s in 1 2 3 4 5; do
    TAG="CLAP_SAM_${s}"
    echo "  Seed $s at $(date)"
    CUDA_VISIBLE_DEVICES=$GPU python main.py \
        --tag "$TAG" --dataset icbhi --seed "$s" \
        --data_folder ./data/ --soft_label_mode "none" \
        --class_split lungsound --n_cls 4 --epochs 50 --batch_size 8 \
        --optimizer adam --learning_rate 2e-5 --weight_decay 1e-6 \
        --cosine --sample_rate 48000 \
        --model laion/clap-htsat-unfused --model_type ClapModel \
        --meta_mode all --test_fold official --pad_types repeat \
        --ma_update --ma_beta 0.5 --method ce \
        --num_workers 4 --print_freq 100 \
        --focal_loss --focal_gamma 2.0 --focal_alpha auto \
        --sam --sam_rho 0.03 --label_smoothing 0.05 \
        2>&1 | tee "$LOGDIR/clap_sam_${s}.log"
done

# 1c. AST + balanced training (5 seeds)
echo "[1c] AST + balanced training..."
for s in 1 2 3 4 5; do
    TAG="AST_balanced_${s}"
    echo "  Seed $s at $(date)"
    CUDA_VISIBLE_DEVICES=$GPU python main.py \
        --tag "$TAG" --dataset icbhi --seed "$s" \
        --data_folder ./data/ --soft_label_mode "none" \
        --class_split lungsound --n_cls 4 --epochs 60 --batch_size 16 \
        --optimizer adam --learning_rate 1e-5 --weight_decay 1e-4 \
        --cosine --warm --warm_epochs 5 --sample_rate 16000 \
        --model ast --from_sl_official --audioset_pretrained \
        --test_fold official --num_workers 4 --print_freq 100 \
        --specaug_policy icbhi_ast_sup \
        --focal_loss --focal_gamma 3.0 --focal_alpha auto \
        --class_balanced_sampling \
        --mixup 0.3 --label_smoothing 0.15 \
        --sam --sam_rho 0.05 \
        2>&1 | tee "$LOGDIR/ast_balanced_${s}.log"
done

# ═══════════════════════════════════════════════
# Phase 2: Extract Logits
# ═══════════════════════════════════════════════
echo ""
echo "═══ Phase 2: Extracting Logits ═══"

# Extract from best BEATs+SAM seeds
mkdir -p teacher_logits_beats_sam
for s in 1 2 3 4 5; do
    CKPT="save/icbhi_beats_ce_BEATs_SAM_${s}/best.pth"
    if [ -f "$CKPT" ]; then
        echo "  Extracting BEATs_SAM_${s}"
        CUDA_VISIBLE_DEVICES=$GPU python scripts/extract_logits.py \
            --checkpoint "$CKPT" --model beats --n_cls 4 \
            --output_dir "teacher_logits_beats_sam/seed_${s}" --num_workers 4
    fi
done

# Extract from best CLAP+SAM seeds
mkdir -p teacher_logits_clap_sam
for s in 1 2 3 4 5; do
    CKPT="save/icbhi_laion/clap-htsat-unfused_ce_all_CLAP_SAM_${s}/best.pth"
    if [ -f "$CKPT" ]; then
        echo "  Extracting CLAP_SAM_${s}"
        CUDA_VISIBLE_DEVICES=$GPU python scripts/extract_logits.py \
            --checkpoint "$CKPT" --model laion/clap-htsat-unfused --n_cls 4 \
            --output_dir "teacher_logits_clap_sam/seed_${s}" --num_workers 4
    fi
done

# Extract from best AST balanced seeds
mkdir -p teacher_logits_ast_balanced
for s in 1 2 3 4 5; do
    CKPT="save/icbhi_ast_ce_AST_balanced_${s}/best.pth"
    if [ -f "$CKPT" ]; then
        echo "  Extracting AST_balanced_${s}"
        CUDA_VISIBLE_DEVICES=$GPU python scripts/extract_logits.py \
            --checkpoint "$CKPT" --model ast --n_cls 4 \
            --output_dir "teacher_logits_ast_balanced/seed_${s}" --num_workers 4
    fi
done

# ═══════════════════════════════════════════════
# Phase 3: Ensemble + TTA + Threshold
# ═══════════════════════════════════════════════
echo ""
echo "═══ Phase 3: Ensemble Evaluation ═══"

# Collect all logits directories
LOGITS_DIRS="teacher_logits teacher_logits_larger_clap teacher_logits_fused teacher_logits_clap_sam"
LOGITS_BEATS="teacher_logits_beats teacher_logits_beats_sam"

echo "Running multi-model ensemble..."
CUDA_VISIBLE_DEVICES=$GPU python scripts/ensemble_multi_model.py \
    --logits_dirs $LOGITS_DIRS \
    --logits_dirs_beats $LOGITS_BEATS \
    --n_cls 4 --grid_steps 30 \
    2>&1 | tee "$LOGDIR/ensemble_results.log"

echo ""
echo "Running threshold optimization..."
CUDA_VISIBLE_DEVICES=$GPU python scripts/threshold_opt.py \
    --logits_dir $LOGITS_DIRS $LOGITS_BEATS \
    --n_cls 4 --grid_steps 30 \
    2>&1 | tee "$LOGDIR/threshold_results.log"

echo ""
echo "Running stacking ensemble..."
CUDA_VISIBLE_DEVICES=$GPU python scripts/stacking_ensemble.py \
    --logits_dirs $LOGITS_DIRS \
    --logits_dirs_beats $LOGITS_BEATS \
    --n_cls 4 --cv_folds 5 \
    2>&1 | tee "$LOGDIR/stacking_results.log"

# ═══════════════════════════════════════════════
# Phase 4: CACD Distillation to CNN14
# ═══════════════════════════════════════════════
echo ""
echo "═══ Phase 4: CACD Distillation to CNN14 ═══"

# Create ensemble soft labels for distillation
python3 -c "
import torch, numpy as np, os

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

# Load all logits
dirs = ['teacher_logits', 'teacher_logits_larger_clap', 'teacher_logits_fused', 'teacher_logits_clap_sam']
beats_dirs = ['teacher_logits_beats', 'teacher_logits_beats_sam']

all_train = []
all_test = []
for d in dirs + beats_dirs:
    tr = os.path.join(d, 'teacher_logits.training.pt')
    te = os.path.join(d, 'teacher_logits.test.pt')
    if os.path.exists(tr) and os.path.exists(te):
        tr_logits = torch.load(tr, weights_only=False)
        te_logits = torch.load(te, weights_only=False)
        if isinstance(tr_logits, list): tr_logits = np.array(tr_logits)
        if isinstance(te_logits, list): te_logits = np.array(te_logits)
        if isinstance(tr_logits, torch.Tensor): tr_logits = tr_logits.numpy()
        if isinstance(te_logits, torch.Tensor): te_logits = te_logits.numpy()
        if tr_logits.ndim == 3: tr_logits = tr_logits.mean(axis=1)
        if te_logits.ndim == 3: te_logits = te_logits.mean(axis=1)
        all_train.append(tr_logits)
        all_test.append(te_logits)
        print(f'  Loaded: {d}')

# Average all logits
train_avg = np.mean(all_train, axis=0)
test_avg = np.mean(all_test, axis=0)

# Reshape to [N, 1, 4] for soft_label_mode='mean_1'
train_reshaped = train_avg.reshape(-1, 1, 4)
test_reshaped = test_avg.reshape(-1, 1, 4)

os.makedirs('teacher_logits_sota_ensemble', exist_ok=True)
torch.save(train_reshaped, 'teacher_logits_sota_ensemble/teacher_logits.training.pt')
torch.save(test_reshaped, 'teacher_logits_sota_ensemble/teacher_logits.test.pt')
print(f'Saved: train={train_reshaped.shape}, test={test_reshaped.shape}')
"

echo "Running CACD distillation..."
CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --tag "CACD_SOTA_CNN14" --dataset icbhi --seed 1 \
    --data_folder ./data/ \
    --soft_label_mode mean_1 \
    --train_teacher_logits teacher_logits_sota_ensemble/teacher_logits.training.pt \
    --test_teacher_logits teacher_logits_sota_ensemble/teacher_logits.test.pt \
    --class_split lungsound --n_cls 4 --epochs 60 --batch_size 32 \
    --optimizer adam --learning_rate 1e-4 --weight_decay 1e-4 \
    --cosine --warm --warm_epochs 5 --sample_rate 16000 \
    --model cnn14 --test_fold official --num_workers 4 --print_freq 50 \
    --cacd --cacd_T 4.0 --cacd_beta 0.5 --cacd_alpha 0.5 \
    --cacd_feat_weight 0.15 --cacd_stage1_epochs 10 \
    --focal_loss --focal_gamma 2.0 --focal_alpha auto \
    --sam --sam_rho 0.03 --label_smoothing 0.1 \
    2>&1 | tee "$LOGDIR/cacd_distill.log"

# ═══════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════
echo ""
echo "============================================"
echo " Pipeline Complete!"
echo " Results: save/results.json"
echo " Ensemble: save/ensemble_results.json"
echo " Thresholds: save/optimal_thresholds.npy"
echo " Stacking: save/stacking_meta_learner.pkl"
echo " Logs: $LOGDIR/"
echo " End: $(date)"
echo "============================================"
