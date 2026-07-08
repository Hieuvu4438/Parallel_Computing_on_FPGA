#!/bin/bash
# Extract logits from BEATs ensemble (top 5 seeds) and save for distillation
# Then train CNN14 student with those soft labels

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export LD_LIBRARY_PATH=/home/haipd/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd /home/haipd/Parallel_Computing_on_FPGA/python/rsc-ensemble-kd

echo "=== Step 1: Extract BEATs ensemble logits ==="
taskset -c 0-7 python -u -c "
import torch
import sys
import os
import numpy as np
sys.path.insert(0, '.')

from models.beats import BEATs
from util.icbhi_dataset import ICBHINonBTSDataset
from torchvision import transforms
import glob

# Top 5 seeds based on results
TOP_SEEDS = [15, 6, 10, 2, 11]
# Or use all 20 seeds:
# TOP_SEEDS = list(range(1, 21))

print(f'Using {len(TOP_SEEDS)} BEATs teachers: {TOP_SEEDS}')

# Find save directories
save_dirs = {}
for seed in TOP_SEEDS:
    patterns = [
        f'save/icbhi_beats_ce_BEATs_ensemble_{seed}',
        f'save/icbhi_beats_ce_BEATs_teacher_{seed}',
    ]
    for p in patterns:
        if os.path.exists(p):
            save_dirs[seed] = p
            break

print(f'Found save dirs: {list(save_dirs.keys())}')

# Args for dataset
class Args:
    dataset = 'icbhi'
    data_folder = './data/'
    class_split = 'lungsound'
    n_cls = 4
    test_fold = 'official'
    sample_rate = 16000
    desired_length = 8
    pad_types = 'repeat'
    model = 'beats'
    soft_label_mode = 'none'
    train_teacher_logits = 'teacher_logits/teacher_logits.training.pt'
    test_teacher_logits = 'teacher_logits/teacher_logits.test.pt'
    specaug_policy = 'icbhi_ast_sup'
    specaug_mask = 'mean'
    resz = 1.0
    n_mels = 128

args = Args()
args.h = int(args.desired_length * 100 - 2)
args.w = 128

# Load datasets
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))
])

print('Loading datasets...')
train_dataset = ICBHINonBTSDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
val_dataset = ICBHINonBTSDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)

# Extract logits from each teacher
def extract_logits(dataset, model):
    model.eval()
    logits_list = []
    with torch.no_grad():
        for i in range(len(dataset)):
            audio = dataset[i][0].unsqueeze(0).cuda()
            features = model(audio, args=args, training=False)
            logit = model.classify(features)
            logits_list.append(logit.cpu().numpy()[0])
            if (i+1) % 500 == 0:
                print(f'  Processed {i+1}/{len(dataset)}')
    return np.array(logits_list)

# Process each split
for split, dataset in [('training', train_dataset), ('test', val_dataset)]:
    print(f'\n=== Processing {split} set ({len(dataset)} samples) ===')

    all_logits = []
    for seed in TOP_SEEDS:
        save_dir = save_dirs[seed]
        ckpt_path = os.path.join(save_dir, 'best.pth')

        print(f'  Loading BEATs seed {seed} from {ckpt_path}')
        model = BEATs(label_dim=4, freeze_encoder=False)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        model.cuda()

        logits = extract_logits(dataset, model)
        all_logits.append(logits)
        print(f'  Seed {seed}: logits shape {logits.shape}')

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Stack and save: [n_samples, n_teachers, n_classes]
    all_logits = np.stack(all_logits, axis=1)
    print(f'  Combined logits shape: {all_logits.shape}')

    os.makedirs('teacher_logits_beats', exist_ok=True)
    save_path = f'teacher_logits_beats/teacher_logits.{split}.pt'
    torch.save(all_logits.tolist(), save_path)
    print(f'  Saved to {save_path}')

print('\n=== Step 1 DONE ===')
" 2>&1 | tee logs/extract_BEATs_logits.log

echo ""
echo "=== Step 2: Train CNN14 student with BEATs ensemble soft labels ==="
taskset -c 0-7 python -u main.py --tag CNN14_from_BEATs_ensemble \
    --dataset icbhi --seed 1 --data_folder ./data/ \
    --soft_label_mode "mean" \
    --train_teacher_logits teacher_logits_beats/teacher_logits.training.pt \
    --test_teacher_logits teacher_logits_beats/teacher_logits.test.pt \
    --class_split lungsound --n_cls 4 \
    --epochs 50 --batch_size 32 --optimizer adam --learning_rate 5e-6 \
    --weight_decay 1e-4 --cosine --sample_rate 16000 --model cnn14 \
    --test_fold official --pad_types repeat \
    --method ce --num_workers 0 --print_freq 50 \
    --focal_loss --focal_gamma 2.0 --focal_alpha auto \
    2>&1 | tee logs/distill_BEATs_to_CNN14.log

echo "=== DISTILLATION DONE ==="
