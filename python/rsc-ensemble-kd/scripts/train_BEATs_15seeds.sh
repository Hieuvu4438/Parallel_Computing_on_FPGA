#!/bin/bash
# Train BEATs seeds 6-20, 2-3 in parallel
# Each model uses ~6-7GB VRAM, 2-3 parallel = ~14-21GB (fits in 48GB)
# CPU: ~80-100% each, 2-3 parallel = ~160-300% (user limit: ~200%)

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export LD_LIBRARY_PATH=/home/haipd/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd /home/haipd/Parallel_Computing_on_FPGA/python/rsc-ensemble-kd

# Wait for seeds 4 and 5 to finish
echo "Waiting for seeds 4 and 5 to finish..."
while true; do
    s4_done=$(grep -c "finished" logs/train_BEATs_seed4.log 2>/dev/null)
    s5_done=$(grep -c "finished" logs/train_BEATs_seed5.log 2>/dev/null)
    if [ "$s4_done" -gt 0 ] && [ "$s5_done" -gt 0 ]; then
        echo "Seeds 4 and 5 finished!"
        break
    fi
    sleep 10
done

# Function to start a seed
start_seed() {
    local seed=$1
    local cpu_start=$2
    local cpu_end=$3
    local tag="BEATs_ensemble_${seed}"
    echo "Starting seed $seed (CPU cores $cpu_start-$cpu_end)..."
    taskset -c ${cpu_start}-${cpu_end} python -u main.py --tag $tag \
        --dataset icbhi --seed $seed --data_folder ./data/ \
        --soft_label_mode none --class_split lungsound --n_cls 4 \
        --epochs 50 --batch_size 8 --optimizer adam --learning_rate 5e-6 \
        --weight_decay 1e-4 --cosine --warm --sample_rate 16000 --model beats \
        --test_fold official --pad_types repeat \
        --method ce --num_workers 0 --print_freq 50 \
        2>&1 | tee logs/train_BEATs_seed${seed}.log &
    echo "  PID: $!"
}

# Run seeds 6-20 in batches of 2
SEEDS=(6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
BATCH_SIZE=2

for ((i=0; i<${#SEEDS[@]}; i+=BATCH_SIZE)); do
    batch=(${SEEDS[@]:i:BATCH_SIZE})
    echo ""
    echo "=== Starting batch: ${batch[@]} ==="

    # Start batch
    for j in "${!batch[@]}"; do
        seed=${batch[$j]}
        cpu_start=$((j * 8))
        cpu_end=$((cpu_start + 7))
        start_seed $seed $cpu_start $cpu_end
    done

    # Wait for batch to finish
    echo "Waiting for batch ${batch[@]} to finish..."
    for seed in "${batch[@]}"; do
        while true; do
            done_flag=$(grep -c "finished" logs/train_BEATs_seed${seed}.log 2>/dev/null)
            if [ "$done_flag" -gt 0 ]; then
                echo "Seed $seed finished!"
                break
            fi
            sleep 30
        done
    done

    echo "=== Batch ${batch[@]} done ==="
done

echo ""
echo "=== ALL 15 SEEDS FINISHED ==="
echo "Results:"
for seed in "${SEEDS[@]}"; do
    result=$(grep "best Score" logs/train_BEATs_seed${seed}.log 2>/dev/null | tail -1)
    echo "  Seed $seed: $result"
done
