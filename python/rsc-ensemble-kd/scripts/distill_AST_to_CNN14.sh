#!/bin/bash
# Distill AST AudioSet-pretrained teacher to CNN14 student
# Uses soft labels from trained AST teacher
# CPU-limited: taskset + OMP_NUM_THREADS + num_workers=0

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export LD_LIBRARY_PATH=/home/haipd/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd /home/haipd/Parallel_Computing_on_FPGA/python/rsc-ensemble-kd

# First, extract logits from AST teacher
echo "=== Step 1: Extract AST teacher logits ==="
taskset -c 0-7 python -u -c "
import torch
import sys
sys.path.insert(0, '.')
from models.ast import ASTModel
from util.icbhi_dataset import ICBHINonBTSDataset
from torchvision import transforms
import argparse

class Args:
    dataset = 'icbhi'
    data_folder = './data/'
    class_split = 'lungsound'
    n_cls = 4
    test_fold = 'official'
    sample_rate = 16000
    desired_length = 8
    pad_types = 'repeat'
    model = 'ast'
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

# Load AST teacher
print('Loading AST AudioSet-pretrained teacher...')
model = ASTModel(
    input_fdim=int(args.h * args.resz),
    input_tdim=int(args.w * args.resz),
    label_dim=args.n_cls,
    imagenet_pretrain=True,
    audioset_pretrain=True,
)

# Load best checkpoint
import glob
save_dirs = glob.glob('save/icbhi_ast_ce_AST_AudioSet_teacher_1*')
if save_dirs:
    ckpt_path = f'{save_dirs[0]}/best.pth'
    print(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
else:
    print('Warning: No AST checkpoint found, using pretrained weights')

model.eval()
model.cuda()

# Load datasets
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))
])

train_dataset = ICBHINonBTSDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
val_dataset = ICBHINonBTSDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)

# Extract logits
def extract_logits(dataset, model):
    logits_list = []
    with torch.no_grad():
        for i in range(len(dataset)):
            audio, label = dataset[i][0], dataset[i][1][0]
            audio = audio.unsqueeze(0).cuda()
            features = model(audio, args=args, training=False)
            logit = model.mlp_head(features)
            logits_list.append(logit.cpu().numpy())
            if (i+1) % 500 == 0:
                print(f'  Processed {i+1}/{len(dataset)}')
    return logits_list

print('Extracting train logits...')
train_logits = extract_logits(train_dataset, model)
print('Extracting test logits...')
test_logits = extract_logits(val_dataset, model)

# Save as teacher logits format [N, 1, 4] (1 teacher)
import os
os.makedirs('teacher_logits_ast', exist_ok=True)
torch.save(train_logits, 'teacher_logits_ast/teacher_logits.training.pt')
torch.save(test_logits, 'teacher_logits_ast/teacher_logits.test.pt')
print(f'Saved teacher logits to teacher_logits_ast/')
"

echo "=== Step 2: Distill to CNN14 ==="
taskset -c 0-7 python -u main.py --tag CNN14_from_AST \
    --dataset icbhi --seed 1 --data_folder ./data/ \
    --soft_label_mode mean_1 --class_split lungsound --n_cls 4 \
    --train_teacher_logits teacher_logits_ast/teacher_logits.training.pt \
    --test_teacher_logits teacher_logits_ast/teacher_logits.test.pt \
    --epochs 50 --batch_size 32 --optimizer adam --learning_rate 5e-6 \
    --weight_decay 1e-4 --cosine --sample_rate 16000 --model cnn14 \
    --test_fold official --pad_types repeat \
    --method ce --num_workers 0 --print_freq 50 2>&1 | tee logs/distill_AST_CNN14.log

echo "=== DISTILLATION DONE ==="
