#!/bin/bash
# Ablation: Proper KD only (temperature + alpha blending)
# Tests KD loss with temperature scaling against baseline soft-label CE.

MODEL="laion/clap-htsat-unfused"
SEED="1 2 3 4 5"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="BTS_kd_${s}"
        CUDA_VISIBLE_DEVICES=0 python main.py --tag $TAG \
                                        --dataset icbhi \
                                        --seed $s \
                                        --data_folder ./data/ \
                                        --soft_label_mode "mean_5" \
                                        --class_split lungsound \
                                        --n_cls 4 \
                                        --epochs 50 \
                                        --batch_size 8 \
                                        --optimizer adam \
                                        --learning_rate 5e-5 \
                                        --weight_decay 1e-6 \
                                        --cosine \
                                        --sample_rate 48000 \
                                        --model $m \
                                        --model_type ClapModel \
                                        --meta_mode all \
                                        --test_fold official \
                                        --pad_types repeat \
                                        --ma_update \
                                        --ma_beta 0.5 \
                                        --method ce \
                                        --print_freq 100 \
                                        --kd_temperature 4.0 \
                                        --kd_alpha 0.5
    done
done
