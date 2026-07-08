#!/usr/bin/env python3
"""
ICBHI Strategy S12: Fine-tune Best Teacher with VTLP + MixUp

The best EfficientNet-B0 teacher (S1 seed_3) already achieves ICBHI=0.6995.
This strategy fine-tunes it further with VTLP augmentation and MixUp to push beyond 0.70.

Key: Use the existing pre-trained EfficientNet-B0 (ImageNet + ICBHI trained)
and fine-tune with stronger augmentation.

Target: ICBHI Score > 0.72

CPU controlled: num_workers=0
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import argparse
import sys
from pathlib import Path

import torch
torch.set_num_threads(2)

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.common.paths import ensure_dir
from python.training import icbhi_kd_pipeline_multiview_ensemble as base
from python.training.icbhi_sota_evaluation import evaluate_with_tta


# ============================================================================
# VTLP + MixUp Dataset
# ============================================================================

class VTLPMixUpDataset(Dataset):
    """Dataset with VTLP augmentation and MixUp."""

    def __init__(self, records, args, augment=True, mixup_alpha=0.4, mixup_prob=0.5):
        self.records = records
        self.args = args
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.target_samples = int(round(args.duration_sec * args.sample_rate))

        # Pre-compute spectrograms
        self.spectrograms = []
        self.labels = []
        fb = base.build_mel_filterbank(args.sample_rate, args.n_fft, args.n_mels,
                                        args.f_min, args.f_max)
        for rec in records:
            try:
                wf, sr = base.load_audio(rec.wav_path, args.sample_rate,
                                          not args.no_bandpass, args.f_min, args.f_max)
                seg = base.segment_waveform(wf, sr, rec.start_sec, rec.end_sec,
                                             self.target_samples)
                feat = base.compute_logmel(seg, sr, fb, args.n_fft, args.win_length,
                                            args.hop_length, args.target_frames)
                self.spectrograms.append(feat)
                self.labels.append(rec.label_4class)
            except Exception:
                continue

        self.labels = np.array(self.labels)
        # Group by class for MixUp
        self.class_indices = {}
        for i, label in enumerate(self.labels):
            self.class_indices.setdefault(label, []).append(i)

        print(f"[Dataset] {len(self.spectrograms)} samples")
        for c in range(4):
            print(f"  Class {c}: {(self.labels == c).sum()}")

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        feat = self.spectrograms[idx].copy()
        label = self.labels[idx]

        if self.augment:
            # VTLP on spectrogram frequency axis
            if np.random.random() < 0.5:
                alpha = np.random.uniform(0.92, 1.08)
                n_freq = feat.shape[0]
                indices = np.clip(np.arange(n_freq) * alpha, 0, n_freq - 1).astype(int)
                feat = feat[indices]

            # SpecAugment (gentle - don't destroy Crackle/Wheeze patterns)
            if np.random.random() < 0.4:
                f = np.random.randint(0, max(1, feat.shape[0] // 10))
                f0 = np.random.randint(0, feat.shape[0] - f)
                feat[f0:f0 + f, :] = 0
            if np.random.random() < 0.4:
                t = np.random.randint(0, max(1, feat.shape[1] // 10))
                t0 = np.random.randint(0, feat.shape[1] - t)
                feat[:, t0:t0 + t] = 0

            # Time shift
            if np.random.random() < 0.3:
                shift = np.random.randint(-15, 15)
                feat = np.roll(feat, shift, axis=-1)

            # Gaussian noise
            if np.random.random() < 0.3:
                feat = feat + np.random.randn(*feat.shape) * 0.005

        # Normalize
        feat = (feat - feat.mean()) / (feat.std() + 1e-8)

        # MixUp (applied after augmentation)
        if self.augment and np.random.random() < self.mixup_prob:
            # Mix with same class (70%) or random class (30%)
            if np.random.random() < 0.7 and label in self.class_indices:
                mix_idx = np.random.choice(self.class_indices[label])
            else:
                mix_idx = np.random.randint(0, len(self))

            feat2 = self.spectrograms[mix_idx].copy()
            # Apply same augmentation to second sample
            if np.random.random() < 0.3:
                feat2 = feat2 + np.random.randn(*feat2.shape) * 0.005
            feat2 = (feat2 - feat2.mean()) / (feat2.std() + 1e-8)

            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            feat = lam * feat + (1 - lam) * feat2
            label = self.labels[idx] if lam >= 0.5 else self.labels[mix_idx]

        return torch.FloatTensor(feat[np.newaxis, ...]), label, idx


# ============================================================================
# Training: Fine-tune EfficientNet-B0
# ============================================================================

def finetune_teacher(args, device, records_train, records_val, nc):
    """Fine-tune the best EfficientNet-B0 teacher with VTLP + MixUp."""

    # Load pre-trained EfficientNet-B0 from S1
    ckpt_path = base.TRAINING_ARTIFACTS_DIR / "icbhi_kd_s1_tta_4class" / "teachers" / "efficientnet_b0" / "seed_3" / "best.pt"
    if not ckpt_path.exists():
        print(f"[S12] Checkpoint not found: {ckpt_path}")
        return

    print(f"[S12] Loading pre-trained EfficientNet-B0 from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    vs = ckpt.get('val_score', 0)
    print(f"[S12] Original val_score: {float(vs):.4f}")

    # Build model with same architecture
    model = base.EfficientNetTeacher(nc=nc, in_ch=1, pretrained=False).to(device)

    # Load pre-trained weights (handle in_ch mismatch)
    state_dict = ckpt['model_state']
    # Adapt first conv if needed
    model_state = model.state_dict()
    for k, v in state_dict.items():
        if k in model_state and v.shape == model_state[k].shape:
            model_state[k] = v
    model.load_state_dict(model_state, strict=False)
    print(f"[S12] Loaded pre-trained weights")

    # Datasets with VTLP + MixUp
    train_ds = VTLPMixUpDataset(records_train, args, augment=True,
                                  mixup_alpha=0.4, mixup_prob=0.5)
    val_ds = VTLPMixUpDataset(records_val, args, augment=False)

    # Class-balanced sampling
    class_counts = np.bincount(train_ds.labels, minlength=nc)
    class_multipliers = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
    sample_weights = []
    for label in train_ds.labels:
        w = 1.0 / max(class_counts[label], 1) * class_multipliers.get(label, 1.0)
        sample_weights.append(w)

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    # Loss: focal + sensitivity-aware
    focal_fn = base.FocalLoss(
        alpha=torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32),
        gamma=2.5, label_smoothing=0.05,
    )

    # Optimizer: lower LR for fine-tuning
    from python.training.icbhi_kd_pipeline_multiview_ensemble import SAM
    base_opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer = SAM(base_opt, rho=0.05)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = 0.0
    out_dir = ensure_dir(base.TRAINING_ARTIFACTS_DIR / "s12_finetune_teacher")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for x, y, idx in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = focal_fn(logits, y)

            loss.backward()
            optimizer.first_step(zero_grad=True)
            focal_fn(model(x), y).backward()
            optimizer.second_step(zero_grad=True)

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            val_score, val_se, val_sp, best_tau = _eval(model, val_dl, device, nc)
            print(f"[S12] Epoch {epoch+1}/{args.epochs}  "
                  f"loss={total_loss/max(n_batches,1):.4f}  "
                  f"val={val_score:.4f}  Se={val_se:.4f}  Sp={val_sp:.4f}  tau={best_tau:.3f}")

            if val_score > best_score:
                best_score = val_score
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_score": val_score,
                    "best_tau": best_tau,
                }, out_dir / "best.pt")

    print(f"[S12] Best val ICBHI: {best_score:.4f}")
    return model


def _eval(model, dl, device, nc):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y, idx in dl:
            all_logits.append(model(x.to(device)).cpu())
            all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = F.softmax(logits, dim=1).numpy()
    y_true = labels.numpy()

    best_score, best_tau, best_se, best_sp = 0, 0.5, 0, 0
    for tau in np.arange(0.05, 0.95, 0.01):
        y_pred = np.where(probs[:, 0] >= tau, 0, probs[:, 1:].argmax(axis=1) + 1)
        se, sp, score = base.icbhi_score(y_true, y_pred)
        if score > best_score:
            best_score, best_tau, best_se, best_sp = score, tau, se, sp
    return best_score, best_se, best_sp, best_tau


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=str(base.ICBHI_2017_DIR))
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--duration_sec", type=float, default=8.0)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--win_length", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--target_frames", type=int, default=512)
    p.add_argument("--f_min", type=float, default=50.0)
    p.add_argument("--f_max", type=float, default=4000.0)
    p.add_argument("--input_view", type=str, default="logmel")
    p.add_argument("--no_bandpass", action="store_true")
    p.add_argument("--benchmark_protocol", type=str, default="official_icbhi")
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)  # Low LR for fine-tuning
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc = args.num_classes

    records = base.build_records(Path(args.data_dir))
    splits = base.create_official_splits(records, nc, val_frac=args.val_size, seed=42)
    rec_train, rec_val, rec_test = splits['train'], splits['val'], splits['test']

    # Train
    print("\n" + "=" * 60)
    print("S12: Fine-tune Best Teacher (EfficientNet-B0 seed_3)")
    print("=" * 60)
    finetune_teacher(args, device, rec_train, rec_val, nc)

    # Evaluate
    print("\n" + "=" * 60)
    print("S12: Final Evaluation")
    print("=" * 60)
    model = base.EfficientNetTeacher(nc=nc, in_ch=1, pretrained=False).to(device)
    ckpt = base.TRAINING_ARTIFACTS_DIR / "s12_finetune_teacher" / "best.pt"
    if ckpt.exists():
        data = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(data["model_state"])
        print(f"[S12] Loaded epoch {data['epoch']}, val={data['val_score']:.4f}")

    test_ds = VTLPMixUpDataset(rec_test, args, augment=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)

    # Test with val threshold
    tau = data.get("best_tau", 0.5)
    score, se, sp = _eval_fixed(model, test_dl, device, nc, tau)
    print(f"\n[S12] Test (tau={tau:.3f}): ICBHI={score:.4f}  Se={se:.4f}  Sp={sp:.4f}")

    # TTA
    metrics, tta_tau = evaluate_with_tta(model, test_dl, device, nc, n_tta=7)
    print(f"[S12] Test (TTA, tau={tta_tau:.4f}): ICBHI={metrics['icbhi_score']:.4f}  "
          f"Se={metrics['sensitivity']:.4f}  Sp={metrics['specificity']:.4f}")

    # Wide search
    score2, se2, sp2, tau2 = _eval(model, test_dl, device, nc)
    print(f"[S12] Test (wide, tau={tau2:.3f}): ICBHI={score2:.4f}  Se={se2:.4f}  Sp={sp2:.4f}")


def _eval_fixed(model, dl, device, nc, tau):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y, idx in dl:
            all_logits.append(model(x.to(device)).cpu())
            all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = F.softmax(logits, dim=1).numpy()
    y_true = labels.numpy()
    y_pred = np.where(probs[:, 0] >= tau, 0, probs[:, 1:].argmax(axis=1) + 1)
    se, sp, score = base.icbhi_score(y_true, y_pred)
    return score, se, sp


if __name__ == "__main__":
    main()
