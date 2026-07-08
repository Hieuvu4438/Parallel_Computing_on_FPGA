#!/bin/bash
# Quick Multi-Model Ensemble — Uses existing logits, no retraining needed.
# Evaluates ensemble of CLAP unfused + BEATs with optimal weights.
#
# Usage: bash scripts/run_quick_ensemble.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "============================================"
echo " Quick Ensemble Evaluation"
echo " Using existing logits (no retraining)"
echo "============================================"

# Check what logits are available
echo ""
echo "Available logits:"
for d in teacher_logits teacher_logits_beats teacher_logits_larger_clap teacher_logits_fused; do
    if [ -d "$d" ]; then
        count=$(ls "$d"/*.pt 2>/dev/null | wc -l)
        echo "  ✓ $d/ ($count files)"
    else
        echo "  ✗ $d/ (not found)"
    fi
done

# Build logits_dirs argument dynamically
CLAP_DIRS=""
BEATS_DIRS=""

if [ -d "teacher_logits" ]; then
    CLAP_DIRS="teacher_logits"
fi
if [ -d "teacher_logits_larger_clap" ]; then
    CLAP_DIRS="$CLAP_DIRS teacher_logits_larger_clap"
fi
if [ -d "teacher_logits_fused" ]; then
    CLAP_DIRS="$CLAP_DIRS teacher_logits_fused"
fi

if [ -d "teacher_logits_beats" ]; then
    BEATS_DIRS="teacher_logits_beats"
fi

echo ""
echo "Running ensemble with:"
echo "  CLAP dirs: $CLAP_DIRS"
echo "  BEATs dirs: $BEATS_DIRS"

python scripts/ensemble_multi_model.py \
    --logits_dirs $CLAP_DIRS \
    --logits_dirs_beats $BEATS_DIRS \
    --n_cls 4 \
    --grid_steps 30
