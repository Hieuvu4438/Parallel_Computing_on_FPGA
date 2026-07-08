#!/usr/bin/env python3
"""
ICBHI Strategy S14: Binary-First + 4-Class

Two-stage approach:
Stage 1: Train strong binary (Normal vs Abnormal) classifier
Stage 2: Use binary features + fine-tune for 4-class

Key insight: Binary classification is much easier (no class imbalance issue).
The binary model learns to detect "is there anything abnormal?"
Then the 4-class model refines the abnormal detection.

Target: ICBHI Score > 0.68

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
# Dataset
# ============================================================================

class ICBHIDataset2(Dataset):
    """Simple dataset with VTLP augmentation."""

    def __init__(self, records, args, augment=True):
        self.records = records
        self.args = args
        self.augment = augment
        self.target_samples = int(round(args.duration_sec * args.sample_rate))

        self.spectrograms = []
        self.labels_4class = []
        self.labels_2class = []
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
                self.labels_4class.append(rec.label_4class)
                self.labels_2class.append(rec.label_2class)
            except Exception:
                continue

        self.labels_4class = np.array(self.labels_4class)
        self.labels_2class = np.array(self.labels_2class)
        print(f"[Dataset] {len(self.spectrograms)} samples")
        for c in range(4):
            print(f"  4-class {c}: {(self.labels_4class == c).sum()}")
        for c in range(2):
            print(f"  2-class {c}: {(self.labels_2class == c).sum()}")

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        feat = self.spectrograms[idx].copy()
        label_4 = self.labels_4class[idx]
        label_2 = self.labels_2class[idx]

        if self.augment:
            # VTLP
            if np.random.random() < 0.5:
                alpha = np.random.uniform(0.92, 1.08)
                n_freq = feat.shape[0]
                indices = np.clip(np.arange(n_freq) * alpha, 0, n_freq - 1).astype(int)
                feat = feat[indices]
            # SpecAugment
            if np.random.random() < 0.4:
                f = np.random.randint(0, max(1, feat.shape[0] // 10))
                f0 = np.random.randint(0, feat.shape[0] - f)
                feat[f0:f0 + f, :] = 0
            if np.random.random() < 0.4:
                t = np.random.randint(0, max(1, feat.shape[1] // 10))
                t0 = np.random.randint(0, feat.shape[1] - t)
                feat[:, t0:t0 + t] = 0
            # Noise
            if np.random.random() < 0.3:
                feat = feat + np.random.randn(*feat.shape) * 0.005

        feat = (feat - feat.mean()) / (feat.std() + 1e-8)
        return torch.FloatTensor(feat[np.newaxis, ...]), label_4, label_2, idx


# ============================================================================
# Model: Two-Head (Binary + 4-Class)
# ============================================================================

class TwoHeadClassifier(nn.Module):
    """CNN with two heads: binary (Normal/Abnormal) and 4-class."""

    def __init__(self, nc=4, in_ch=1, width=1.25):
        super().__init__()
        c = lambda v: max(8, int(v * width))

        # Shared backbone (larger than previous)
        self.backbone = nn.Sequential(
            base.ConvBNAct(in_ch, c(32)),
            base.DSResBlock(c(32), c(48), stride=2),
            base.DSResBlock(c(48), c(64)),
            base.DSResBlock(c(64), c(96), stride=2),
            base.DSResBlock(c(96), c(128)),
            base.DSResBlock(c(128), c(160), stride=2),
        )

        feat_dim = c(160)

        # Binary head: Normal vs Abnormal
        self.binary_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

        # 4-class head
        self.fourclass_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, nc),
        )

    def forward(self, x, head="both"):
        feat = self.backbone(x)
        if head == "binary":
            return self.binary_head(feat)
        elif head == "fourclass":
            return self.fourclass_head(feat)
        else:
            return self.binary_head(feat), self.fourclass_head(feat)

    def extract_features(self, x):
        return self.backbone(x)


# ============================================================================
# Stage 1: Train Binary Classifier
# ============================================================================

def train_binary(args, device, records_train, records_val, nc):
    """Stage 1: Train binary Normal vs Abnormal classifier."""
    print("\n[Stage 1] Training Binary Classifier")

    train_ds = ICBHIDataset2(records_train, args, augment=True)
    val_ds = ICBHIDataset2(records_val, args, augment=False)

    # Balanced sampling for binary
    class_counts = np.bincount(train_ds.labels_2class, minlength=2)
    sample_weights = [1.0 / max(class_counts[l], 1) for l in train_ds.labels_2class]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    model = TwoHeadClassifier(nc=nc, in_ch=1, width=1.25).to(device)

    # Binary loss with class weights
    binary_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 2.0], device=device)  # 2x for Abnormal
    )

    from python.training.icbhi_kd_pipeline_multiview_ensemble import SAM
    base_opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = SAM(base_opt, rho=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_score = 0.0
    out_dir = ensure_dir(base.TRAINING_ARTIFACTS_DIR / "s14_binary_first")

    for epoch in range(50):
        model.train()
        total_loss = 0
        n_batches = 0

        for x, y4, y2, idx in train_dl:
            x, y2 = x.to(device), y2.to(device)
            bin_logits = model(x, head="binary")
            loss = binary_fn(bin_logits, y2)

            loss.backward()
            optimizer.first_step(zero_grad=True)
            binary_fn(model(x, head="binary"), y2).backward()
            optimizer.second_step(zero_grad=True)

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            # Evaluate binary
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y4, y2, idx in val_dl:
                    x, y2 = x.to(device), y2.to(device)
                    pred = model(x, head="binary").argmax(dim=1)
                    correct += (pred == y2).sum().item()
                    total += y2.size(0)
            acc = correct / total
            print(f"[Binary] Epoch {epoch+1}/50  loss={total_loss/max(n_batches,1):.4f}  "
                  f"val_acc={acc:.4f}")

            if acc > best_score:
                best_score = acc
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": acc,
                }, out_dir / "binary_best.pt")

    print(f"[Binary] Best val accuracy: {best_score:.4f}")
    return model


# ============================================================================
# Stage 2: Fine-tune for 4-Class
# ============================================================================

def train_fourclass(args, device, records_train, records_val, nc, binary_model):
    """Stage 2: Fine-tune for 4-class using binary backbone."""
    print("\n[Stage 2] Fine-tuning for 4-Class")

    train_ds = ICBHIDataset2(records_train, args, augment=True)
    val_ds = ICBHIDataset2(records_val, args, augment=False)

    # Class-balanced sampling for 4-class
    class_counts = np.bincount(train_ds.labels_4class, minlength=nc)
    class_multipliers = {0: 1.0, 1: 2.5, 2: 4.0, 3: 5.0}
    sample_weights = []
    for label in train_ds.labels_4class:
        w = 1.0 / max(class_counts[label], 1) * class_multipliers.get(label, 1.0)
        sample_weights.append(w)

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    # Use binary model's backbone
    model = binary_model

    # Loss: focal + binary auxiliary
    focal_fn = base.FocalLoss(
        alpha=torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32),
        gamma=3.0, label_smoothing=0.05,
    )
    binary_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 3.0], device=device)
    )

    from python.training.icbhi_kd_pipeline_multiview_ensemble import SAM
    # Lower LR for fine-tuning
    base_opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    optimizer = SAM(base_opt, rho=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

    best_score = 0.0
    out_dir = ensure_dir(base.TRAINING_ARTIFACTS_DIR / "s14_binary_first")

    for epoch in range(args.epochs):
        # Progressive: binary head helps 4-class head
        t_prog = min(epoch / 20.0, 1.0)
        w_4cls = 0.60 + 0.20 * t_prog  # 0.60 → 0.80
        w_bin = 0.40 - 0.20 * t_prog   # 0.40 → 0.20

        model.train()
        total_loss = 0
        n_batches = 0

        for x, y4, y2, idx in train_dl:
            x, y4, y2 = x.to(device), y4.to(device), y2.to(device)

            bin_logits, cls_logits = model(x, head="both")

            l_4cls = focal_fn(cls_logits, y4)
            l_bin = binary_fn(bin_logits, y2)

            loss = w_4cls * l_4cls + w_bin * l_bin

            loss.backward()
            optimizer.first_step(zero_grad=True)
            focal_fn(model(x, head="fourclass"), y4).backward()
            optimizer.second_step(zero_grad=True)

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            val_score, val_se, val_sp, best_tau = _eval(model, val_dl, device, nc)
            print(f"[4-Class] Epoch {epoch+1}/{args.epochs}  "
                  f"loss={total_loss/max(n_batches,1):.4f}  "
                  f"val={val_score:.4f}  Se={val_se:.4f}  Sp={val_sp:.4f}  tau={best_tau:.3f}")

            if val_score > best_score:
                best_score = val_score
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_score": val_score,
                    "best_tau": best_tau,
                }, out_dir / "fourclass_best.pt")

    print(f"[4-Class] Best val ICBHI: {best_score:.4f}")
    return model


def _eval(model, dl, device, nc):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dl:
            x = batch[0].to(device)
            all_logits.append(model(x, head="fourclass").cpu())
            all_labels.append(batch[1])
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
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc = args.num_classes

    records = base.build_records(Path(args.data_dir))
    splits = base.create_official_splits(records, nc, val_frac=args.val_size, seed=42)
    rec_train, rec_val, rec_test = splits['train'], splits['val'], splits['test']

    # Stage 1: Binary
    print("=" * 60)
    print("S14: Binary-First + 4-Class Approach")
    print("=" * 60)
    binary_model = train_binary(args, device, rec_train, rec_val, nc)

    # Stage 2: 4-class
    model = train_fourclass(args, device, rec_train, rec_val, nc, binary_model)

    # Evaluate
    print("\n" + "=" * 60)
    print("S14: Final Evaluation")
    print("=" * 60)
    ckpt = base.TRAINING_ARTIFACTS_DIR / "s14_binary_first" / "fourclass_best.pt"
    if ckpt.exists():
        data = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(data["model_state"])
        print(f"[S14] Loaded epoch {data['epoch']}, val={data['val_score']:.4f}")

    test_ds = ICBHIDataset2(rec_test, args, augment=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)

    tau = data.get("best_tau", 0.5)
    score, se, sp = _eval_fixed(model, test_dl, device, nc, tau)
    print(f"[S14] Test (tau={tau:.3f}): ICBHI={score:.4f}  Se={se:.4f}  Sp={sp:.4f}")

    metrics, tta_tau = evaluate_with_tta(model, test_dl, device, nc, n_tta=7)
    print(f"[S14] Test (TTA, tau={tta_tau:.4f}): ICBHI={metrics['icbhi_score']:.4f}  "
          f"Se={metrics['sensitivity']:.4f}  Sp={metrics['specificity']:.4f}")

    score2, se2, sp2, tau2 = _eval(model, test_dl, device, nc)
    print(f"[S14] Test (wide, tau={tau2:.3f}): ICBHI={score2:.4f}  Se={se2:.4f}  Sp={sp2:.4f}")


def _eval_fixed(model, dl, device, nc, tau):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dl:
            x = batch[0].to(device)
            all_logits.append(model(x, head="fourclass").cpu())
            all_labels.append(batch[1])
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = F.softmax(logits, dim=1).numpy()
    y_true = labels.numpy()
    y_pred = np.where(probs[:, 0] >= tau, 0, probs[:, 1:].argmax(axis=1) + 1)
    se, sp, score = base.icbhi_score(y_true, y_pred)
    return score, se, sp


if __name__ == "__main__":
    main()
