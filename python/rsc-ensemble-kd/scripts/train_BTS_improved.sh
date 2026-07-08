#!/bin/bash
# BTS++ with all improvements: Focal Loss + KD + Mixup + Label Smoothing + SAM
# All changes are training-only — evaluation protocol is UNCHANGED.
#
# Usage: bash scripts/train_BTS_improved.sh
#
# Feature summary:
#   --focal_loss --focal_alpha auto --focal_gamma 2.0  : Focal loss with auto class weights
#   --kd_temperature 4.0 --kd_alpha 0.5                : Proper KD with T=4, alpha=0.5
#   --mixup 0.3                                         : Mixup augmentation
#   --label_smoothing 0.1                               : Label smoothing
#   --sam --sam_rho 0.05                                : SAM optimizer
#
# To use individual features, just add/remove the relevant flags.

MODEL="laion/clap-htsat-unfused"
SEED="1"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="BTS_improved_${s}"
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
                                        --focal_loss \
                                        --focal_gamma 2.0 \
                                        --focal_alpha auto \
                                        --kd_temperature 4.0 \
                                        --kd_alpha 0.5 \
                                        --mixup 0.3 \
                                        --label_smoothing 0.1 \
                                        --sam \
                                        --sam_rho 0.05

    done
done
