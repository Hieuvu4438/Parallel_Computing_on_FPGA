#!/usr/bin/env python3
"""
ICBHI Strategy S13: Ensemble KD from Strong Teachers

Use the 3 pre-trained EfficientNet-B0 teachers (ICBHI: 0.6750, 0.6615, 0.6995)
as an ensemble and distill into a lightweight student.

Key: The teachers are already strong. The student learns from their
combined soft labels + VTLP augmentation.

Target: ICBHI Score > 0.66 for the student (lightweight, FPGA-deployable)

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
# Dataset with Teacher Logits + VTLP
# ============================================================================

class TeacherKDDataset(Dataset):
    """Dataset that loads pre-computed teacher logits + VTLP augmentation."""

    def __init__(self, records, teacher_logits, args, augment=True):
        self.records = records
        self.teacher_logits = teacher_logits  # [N_teachers, N_samples, nc]
        self.args = args
        self.augment = augment
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
        print(f"[TeacherKD] {len(self.spectrograms)} samples, "
              f"{teacher_logits.shape[0]} teachers")

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        feat = self.spectrograms[idx].copy()
        label = self.labels[idx]

        if self.augment:
            # VTLP
            if np.random.random() < 0.5:
                alpha = np.random.uniform(0.92, 1.08)
                n_freq = feat.shape[0]
                indices = np.clip(np.arange(n_freq) * alpha, 0, n_freq - 1).astype(int)
                feat = feat[indices]
            # SpecAugment (gentle)
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

        # Normalize
        feat = (feat - feat.mean()) / (feat.std() + 1e-8)

        # Teacher ensemble: average logits from all teachers
        teacher_probs = F.softmax(torch.FloatTensor(self.teacher_logits[:, idx, :]), dim=1).mean(dim=0)

        return torch.FloatTensor(feat[np.newaxis, ...]), label, teacher_probs, idx


# ============================================================================
# Training: Student KD
# ============================================================================

def train_student_kd(args, device, records_train, records_val, nc):
    """Train student with KD from strong teacher ensemble."""

    # Load teacher logits with sample ID matching
    import json
    logits_dir = base.TRAINING_ARTIFACTS_DIR / "icbhi_kd_s1_tta_4class" / "teacher_logits"

    # Load sample IDs from first teacher (all teachers have same IDs)
    with open(logits_dir / "efficientnet_b0_seed_1_train_sample_ids.json") as f:
        train_ids = json.load(f)
    with open(logits_dir / "efficientnet_b0_seed_1_val_sample_ids.json") as f:
        val_ids = json.load(f)

    # Create ID→index mapping
    train_id_to_idx = {sid: i for i, sid in enumerate(train_ids)}
    val_id_to_idx = {sid: i for i, sid in enumerate(val_ids)}

    # Match records to logits
    train_matched = []
    val_matched = []
    for rec in records_train:
        if rec.sample_id in train_id_to_idx:
            train_matched.append(train_id_to_idx[rec.sample_id])
    for rec in records_val:
        if rec.sample_id in val_id_to_idx:
            val_matched.append(val_id_to_idx[rec.sample_id])

    print(f"[S13] Matched: {len(train_matched)}/{len(records_train)} train, "
          f"{len(val_matched)}/{len(records_val)} val")

    # Load and align teacher logits
    teacher_logits_train = []
    teacher_logits_val = []
    for seed in [1, 2, 3]:
        t = np.load(logits_dir / f"efficientnet_b0_seed_{seed}_train.npy")
        v = np.load(logits_dir / f"efficientnet_b0_seed_{seed}_val.npy")
        teacher_logits_train.append(t[train_matched])
        teacher_logits_val.append(v[val_matched])
        print(f"[S13] Loaded teacher: efficientnet_b0 seed={seed}")

    teacher_logits_train = np.stack(teacher_logits_train)  # [3, N_matched, nc]
    teacher_logits_val = np.stack(teacher_logits_val)

    # Filter records to matched only
    records_train = [r for i, r in enumerate(records_train) if r.sample_id in train_id_to_idx]
    records_val = [r for i, r in enumerate(records_val) if r.sample_id in val_id_to_idx]

    # Datasets
    train_ds = TeacherKDDataset(records_train, teacher_logits_train, args, augment=True)
    val_ds = TeacherKDDataset(records_val, teacher_logits_val, args, augment=False)

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

    # Student model (lightweight, FPGA-deployable)
    student = base.DSCNNResSEStudent(nc=nc, in_ch=1, width=1.25).to(device)
    total_params = sum(p.numel() for p in student.parameters())
    print(f"[S13] Student: {total_params/1e6:.2f}M params")

    # Loss: focal + smoothed KD + binary
    focal_fn = base.FocalLoss(
        alpha=torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32),
        gamma=2.5, label_smoothing=0.05,
    )

    T = args.temperature
    kd_smoothing = 0.15

    from python.training.icbhi_kd_pipeline_multiview_ensemble import SAM
    base_opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer = SAM(base_opt, rho=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

    best_score = 0.0
    out_dir = ensure_dir(base.TRAINING_ARTIFACTS_DIR / "s13_ensemble_kd")

    for epoch in range(args.epochs):
        # Progressive weighting: hard→KD shift
        t_prog = min(epoch / 30.0, 1.0)
        w_hard = 0.40 * (1 - t_prog) + 0.25 * t_prog
        w_kd = 0.35 * (1 - t_prog) + 0.50 * t_prog
        w_bin = 0.25 * (1 - t_prog) + 0.25 * t_prog

        student.train()
        total_loss = 0
        n_batches = 0

        for x, y, t_probs, idx in train_dl:
            x, y = x.to(device), y.to(device)
            t_probs = t_probs.to(device)

            logits = student(x)

            # Focal loss
            l_hard = focal_fn(logits, y)

            # Smoothed KD loss
            nc_logits = logits.size(1)
            smoothed_teacher = (1 - kd_smoothing) * t_probs + kd_smoothing / nc_logits
            l_kd = -(smoothed_teacher * F.log_softmax(logits / T, dim=1)
                     ).sum(dim=1).mean() * (T ** 2)

            # Binary auxiliary loss
            abnormal_logit = torch.logsumexp(logits[:, 1:], dim=1) - logits[:, 0]
            bin_target = (y != 0).float()
            teacher_bin = (1.0 - t_probs[:, 0]).clamp(0, 1)
            bin_target_blend = 0.6 * bin_target + 0.4 * teacher_bin
            l_bin = F.binary_cross_entropy_with_logits(abnormal_logit, bin_target_blend)

            loss = w_hard * l_hard + w_kd * l_kd + w_bin * l_bin

            loss.backward()
            optimizer.first_step(zero_grad=True)
            focal_fn(student(x), y).backward()
            optimizer.second_step(zero_grad=True)

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            val_score, val_se, val_sp, best_tau = _eval(student, val_dl, device, nc)
            print(f"[S13] Epoch {epoch+1}/{args.epochs}  "
                  f"loss={total_loss/max(n_batches,1):.4f}  "
                  f"val={val_score:.4f}  Se={val_se:.4f}  Sp={val_sp:.4f}  tau={best_tau:.3f}")

            if val_score > best_score:
                best_score = val_score
                torch.save({
                    "model_state": student.state_dict(),
                    "epoch": epoch,
                    "val_score": val_score,
                    "best_tau": best_tau,
                }, out_dir / "best.pt")

    print(f"[S13] Best val ICBHI: {best_score:.4f}")
    return student


def _eval(model, dl, device, nc):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dl:
            x = batch[0].to(device)
            all_logits.append(model(x).cpu())
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
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=4.0)
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
    print("S13: Ensemble KD from EfficientNet-B0 Teachers")
    print("=" * 60)
    train_student_kd(args, device, rec_train, rec_val, nc)

    # Evaluate
    print("\n" + "=" * 60)
    print("S13: Final Evaluation")
    print("=" * 60)
    student = base.DSCNNResSEStudent(nc=nc, in_ch=1, width=1.25).to(device)
    ckpt = base.TRAINING_ARTIFACTS_DIR / "s13_ensemble_kd" / "best.pt"
    if ckpt.exists():
        data = torch.load(ckpt, map_location=device, weights_only=False)
        student.load_state_dict(data["model_state"])
        print(f"[S13] Loaded epoch {data['epoch']}, val={data['val_score']:.4f}")

    # Load test teacher logits
    test_logits = []
    for seed in [1, 2, 3]:
        test_logits.append(np.load(logits_dir / f"efficientnet_b0_seed_{seed}_test.npy"))
    teacher_logits_test = np.stack(test_logits)

    test_ds = TeacherKDDataset(rec_test, teacher_logits_test, args, augment=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)

    tau = data.get("best_tau", 0.5)
    metrics, tta_tau = evaluate_with_tta(student, test_dl, device, nc, n_tta=7)
    print(f"[S13] Test (TTA, tau={tta_tau:.4f}): ICBHI={metrics['icbhi_score']:.4f}  "
          f"Se={metrics['sensitivity']:.4f}  Sp={metrics['specificity']:.4f}")

    score, se, sp, tau2 = _eval(student, test_dl, device, nc)
    print(f"[S13] Test (wide, tau={tau2:.3f}): ICBHI={score:.4f}  Se={se:.4f}  Sp={sp:.4f}")


if __name__ == "__main__":
    main()
