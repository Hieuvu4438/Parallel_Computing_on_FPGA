#!/bin/bash
# Train AST teacher ensemble on ICBHI (mel-spectrogram 16kHz)
# These teachers will be used to distill into CNN14 student
# CPU-limited execution

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export LD_LIBRARY_PATH=/home/haipd/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd /home/haipd/Parallel_Computing_on_FPGA/python/rsc-ensemble-kd

MODEL="ast"
SEEDS="1 2 3 4 5"

for s in $SEEDS
do
    TAG="AST_teacher_${s}"
    echo "=== Training AST teacher seed=$s ==="
    taskset -c 0-7 python -u main.py --tag $TAG \
                                        --dataset icbhi \
                                        --seed $s \
                                        --data_folder ./data/ \
                                        --soft_label_mode "none" \
                                        --class_split lungsound \
                                        --n_cls 4 \
                                        --epochs 50 \
                                        --batch_size 32 \
                                        --optimizer adam \
                                        --learning_rate 5e-5 \
                                        --weight_decay 1e-4 \
                                        --cosine \
                                        --warm \
                                        --sample_rate 16000 \
                                        --model $MODEL \
                                        --from_sl_official \
                                        --test_fold official \
                                        --pad_types repeat \
                                        --method ce \
                                        --num_workers 0 \
                                        --print_freq 50 \
                                        --specaug_policy icbhi_ast_sup

    echo "=== AST teacher seed=$s finished ==="
done

echo "=== ALL AST TEACHERS DONE ==="
