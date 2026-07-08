"""
Ensemble → CNN14 Knowledge Distillation (simplified)

Loads cached mel-spectrogram data and pre-computed ensemble soft labels,
then trains CNN14 with combined KD + Focal loss.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/distill_ensemble_to_cnn14.py \
        --temperature 3.0 --kd_alpha 0.7 --epochs 100
"""
import os, sys, json, math, time, random, argparse
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms
from transformers import set_seed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_backbone_class
from util.icbhi_util import get_score


class SimpleDataset(torch.utils.data.Dataset):
    """Dataset from pre-cached .pt files + soft labels."""
    def __init__(self, data_path, soft_labels_path, transform=None):
        raw = torch.load(data_path, weights_only=False)
        self.transform = transform

        # Parse cached data: list of (audio_images, label)
        self.images = []
        self.labels = []
        for item in raw:
            img, label = item[0], item[1]
            if isinstance(img, list):
                img = img[0]
            if isinstance(label, tuple):
                label = label[0]
            self.images.append(img)
            self.labels.append(label)

        # Load soft labels
        sl = torch.load(soft_labels_path, weights_only=False)
        if isinstance(sl, np.ndarray):
            sl = torch.from_numpy(sl).float()
        self.soft_labels = sl

        # Class distribution
        self.class_nums = np.bincount(self.labels, minlength=4).astype(float)
        print(f'  Loaded {len(self)} samples, class_dist={self.class_nums.tolist()}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, (self.labels[idx], self.soft_labels[idx])


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        fl = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            fl = self.alpha[targets] * fl
        return fl.mean()


def update_ema(ema_model, model, beta):
    with torch.no_grad():
        for ep, p in zip(ema_model.parameters(), model.parameters()):
            ep.data.mul_(beta).add_(p.data, alpha=1 - beta)


def compute_icbhi(model, loader, n_cls=4):
    model.eval()
    hits = [0.0] * n_cls
    counts = [0.0] * n_cls
    with torch.no_grad():
        for images, (labels, _) in loader:
            images, labels = images.cuda(), labels.cuda()
            preds = model(images).argmax(1)
            for i in range(labels.shape[0]):
                l = labels[i].item()
                counts[l] += 1
                if preds[i].item() == l:
                    hits[l] += 1
    sp, se, sc = get_score(hits, counts)
    return sp, se, sc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--warm_epochs', type=int, default=5)
    p.add_argument('--temperature', type=float, default=3.0)
    p.add_argument('--kd_alpha', type=float, default=0.7)
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--ema_beta', type=float, default=0.999)
    p.add_argument('--soft_labels_dir', type=str, default='teacher_logits')
    p.add_argument('--soft_labels_T', type=float, default=3.0)
    p.add_argument('--save_dir', type=str, default='./save')
    p.add_argument('--tag', type=str, default='distilled_cnn14_ensemble')
    p.add_argument('--print_freq', type=int, default=50)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--n_cls', type=int, default=4)
    args = p.parse_args()

    save_folder = os.path.join(args.save_dir, args.tag)
    os.makedirs(save_folder, exist_ok=True)

    # Seed
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True; cudnn.benchmark = True; set_seed(args.seed)

    # Transform
    h, w = 798, 128
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((h, w))])

    # Datasets
    train_sl = os.path.join(args.soft_labels_dir, f'ensemble_soft_labels_T{args.soft_labels_T}.training.pt')
    test_sl = os.path.join(args.soft_labels_dir, f'ensemble_soft_labels_T{args.soft_labels_T}.test.pt')

    train_ds = SimpleDataset('data/training_nonbts.pt', train_sl, transform)
    val_ds = SimpleDataset('data/test_nonbts.pt', test_sl, transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # Model
    backbone = get_backbone_class('cnn14')()
    classifier = nn.Linear(backbone.final_feat_dim, args.n_cls)

    # EMA
    ema_backbone = deepcopy(backbone).cuda()
    ema_classifier = deepcopy(classifier).cuda()
    for p in ema_backbone.parameters(): p.requires_grad = False
    for p in ema_classifier.parameters(): p.requires_grad = False

    backbone.cuda(); classifier.cuda()

    # Class weights
    counts = train_ds.class_counts if hasattr(train_ds, 'class_counts') else train_ds.class_nums
    beta = 0.999
    eff = 1.0 - torch.pow(torch.tensor(beta), torch.tensor(counts, dtype=torch.float32))
    alpha = (1.0 - beta) / eff
    alpha = alpha / alpha.sum() * args.n_cls
    alpha = alpha.cuda()
    print(f'Class weights: {alpha.tolist()}')

    focal = FocalLoss(alpha=alpha, gamma=args.focal_gamma)
    params = list(backbone.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    best_score = 0
    best_state = {
        'model': deepcopy(ema_backbone.state_dict()),
        'classifier': deepcopy(ema_classifier.state_dict()),
        'epoch': 0,
        'score': [0, 0, 0],
    }

    print(f'\n{"="*60}')
    print(f' Ensemble → CNN14 Distillation')
    print(f' T={args.temperature}, alpha={args.kd_alpha}, epochs={args.epochs}')
    print(f' Student: CNN14 ({sum(p.numel() for p in backbone.parameters())/1e6:.1f}M)')
    print(f'{"="*60}\n')

    for epoch in range(1, args.epochs + 1):
        backbone.train(); classifier.train()
        losses = []

        # Warmup
        if epoch <= args.warm_epochs:
            lr = args.lr * epoch / args.warm_epochs
            for pg in optimizer.param_groups: pg['lr'] = lr
        else:
            # Cosine annealing
            progress = (epoch - args.warm_epochs) / (args.epochs - args.warm_epochs)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups: pg['lr'] = lr

        t0 = time.time()
        for idx, (images, (hard_labels, soft_labels)) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            hard_labels = hard_labels.cuda(non_blocking=True)
            soft_labels = soft_labels.cuda(non_blocking=True)

            with torch.cuda.amp.autocast():
                output = classifier(backbone(images))

                # KD loss
                T = args.temperature
                log_student = F.log_softmax(output / T, dim=1)
                kd_loss = F.kl_div(log_student, soft_labels, reduction='batchmean') * (T * T)

                # Focal loss
                fl_loss = focal(output, hard_labels)

                loss = args.kd_alpha * kd_loss + (1 - args.kd_alpha) * fl_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # EMA update
            update_ema(ema_backbone, backbone, args.ema_beta)
            update_ema(ema_classifier, classifier, args.ema_beta)

            losses.append(loss.item())

            if (idx + 1) % args.print_freq == 0:
                acc = (output.argmax(1) == hard_labels).float().mean() * 100
                print(f'  [{epoch}][{idx+1}/{len(train_loader)}] loss={np.mean(losses[-50:]):.3f} acc={acc:.1f} lr={lr:.6f}')

        # Validate with EMA
        class EMAStudent(nn.Module):
            def __init__(self, bb, cls):
                super().__init__()
                self.bb = bb; self.cls = cls
            def forward(self, x):
                return self.cls(self.bb(x))

        ema_student = EMAStudent(ema_backbone, ema_classifier).cuda()
        sp, se, sc = compute_icbhi(ema_student, val_loader, args.n_cls)

        dt = time.time() - t0
        print(f'Epoch {epoch}: loss={np.mean(losses):.3f} S_p={sp:.2f} S_e={se:.2f} Score={sc:.2f} ({dt:.0f}s)')

        if sc > best_score and se > 5:
            best_score = sc
            best_state = {
                'model': deepcopy(ema_backbone.state_dict()),
                'classifier': deepcopy(ema_classifier.state_dict()),
                'epoch': epoch,
                'score': [sp, se, sc],
            }
            print(f'  ★ New best: Score={sc:.2f}')

    # Save
    save_path = os.path.join(save_folder, 'best.pth')
    torch.save(best_state, save_path)
    print(f'\n{"="*60}')
    sp, se, sc = best_state['score']
    print(f' Best: S_p={sp:.2f}, S_e={se:.2f}, Score={sc:.2f}')
    print(f' Saved: {save_path}')
    print(f'{"="*60}')

    # Update results.json
    from util.misc import update_json
    update_json(args.tag, best_state['score'], path=os.path.join(args.save_dir, 'results.json'))


if __name__ == '__main__':
    main()
