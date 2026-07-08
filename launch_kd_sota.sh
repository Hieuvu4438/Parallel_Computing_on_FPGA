#!/bin/bash
# Launch all 4 KD strategies with SOTA upgrades
# Waits for GPU to have enough free memory before starting

set -e

MIN_FREE_MB=8000  # Need at least 8GB free for training
CHECK_INTERVAL=60 # Check every 60 seconds

echo "=== KD SOTA Training Launcher ==="
echo "Waiting for GPU to have ${MIN_FREE_MB}MB free memory..."

while true; do
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -z "$FREE_MB" ]; then
        echo "[$(date +%H:%M:%S)] No GPU detected, waiting..."
    elif [ "$FREE_MB" -ge "$MIN_FREE_MB" ]; then
        echo "[$(date +%H:%M:%S)] GPU has ${FREE_MB}MB free. Starting training!"
        break
    else
        echo "[$(date +%H:%M:%S)] GPU has ${FREE_MB}MB free (need ${MIN_FREE_MB}MB). Waiting..."
    fi
    sleep $CHECK_INTERVAL
done

cd /home/haipd/Parallel_Computing_on_FPGA

# Strategy 1: TTA + SAM + Lungmix
echo ""
echo "=== Launching S1: TTA + SAM + Lungmix ==="
tmux new-session -d -s kd1 \
    "python python/training/icbhi_kd_s1_tta_calibrated.py 2>&1 | tee logs/kd1_sota.log"

# Strategy 2: Feature Attention (fixed) + SAM
echo "=== Launching S2: Feature Attention + SAM ==="
tmux new-session -d -s kd2 \
    "python python/training/icbhi_kd_s2_feature_attention.py 2>&1 | tee logs/kd2_sota.log"

# Strategy 3: Curriculum EMA + SAM
echo "=== Launching S3: Curriculum EMA + SAM ==="
tmux new-session -d -s kd3 \
    "python python/training/icbhi_kd_s3_curriculum_ema.py 2>&1 | tee logs/kd3_sota.log"

# Strategy 4: Transformer Mega Ensemble (fixed) + SAM
echo "=== Launching S4: Transformer Mega Ensemble + SAM ==="
tmux new-session -d -s kd4 \
    "python python/training/icbhi_kd_s4_transformer_mega_ensemble.py 2>&1 | tee logs/kd4_sota.log"

echo ""
echo "All 4 strategies launched in tmux sessions: kd1, kd2, kd3, kd4"
echo "Monitor with: tmux attach -t kd1"
echo "Check all: for s in kd1 kd2 kd3 kd4; do tmux capture-pane -t \$s -p -S -5; done"
