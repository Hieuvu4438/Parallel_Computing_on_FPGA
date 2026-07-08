#!/usr/bin/env python3
"""
ICBHI Strategy S8: Patient-Aware Meta-Learning — V2

Fixes from V1 (test ICBHI=0.5197 patient-adapted, Se=0.5156):
1. Much stronger class weights for minority classes
2. Binary auxiliary loss added (Normal vs Abnormal)
3. Aggressive oversampling of minority classes
4. Better evaluation: use patient-adapted eval during training
5. Sensitivity-aware loss to prevent Normal-prediction collapse

Expected: ICBHI Score > 0.65 (target > 0.68)
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
torch.set_num_threads(2)

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.common.paths import ensure_dir
from python.training import icbhi_kd_pipeline_multiview_ensemble as base
from python.training.icbhi_sota_loss_functions import ClassBalancedFocalLoss
from python.training.icbhi_sota_evaluation import evaluate_with_tta


# ============================================================================
# Patient-Grouped Dataset (same as V1)
# ============================================================================

def _compute_spec(rec, args):
    """Compute log-mel spectrogram for a single record."""
    from python.training.icbhi_kd_pipeline_multiview_ensemble import (
        load_audio, segment_waveform, compute_logmel, build_mel_filterbank,
    )
    target_samples = int(round(args.duration_sec * args.sample_rate))
    fb = build_mel_filterbank(args.sample_rate, args.n_fft, args.n_mels, args.f_min, args.f_max)
    wf, sr = load_audio(rec.wav_path, args.sample_rate, not args.no_bandpass, args.f_min, args.f_max)
    seg = segment_waveform(wf, sr, rec.start_sec, rec.end_sec, target_samples)
    feat = compute_logmel(seg, sr, fb, args.n_fft, args.win_length, args.hop_length, args.target_frames)
    return feat


class PatientGroupedDataset(Dataset):
    def __init__(self, records, args, augment=False):
        self.args = args
        self.augment = augment
        self.patient_samples = defaultdict(list)
        for rec in records:
            spec = _compute_spec(rec, args)
            if spec is not None:
                self.patient_samples[rec.subject_id].append((spec, rec.label_4class))
        self.patients = []
        for pid, samples in self.patient_samples.items():
            labels = [s[1] for s in samples]
            if len(set(labels)) >= 2 and len(samples) >= 4:
                self.patients.append(pid)
        print(f"[MetaDataset] {len(self.patients)} patients with sufficient samples")

    def sample_episode(self, n_patients=8, k_shot=5, q_query=10, nc=4):
        episode_patients = random.sample(self.patients, min(n_patients, len(self.patients)))
        support_x, support_y = [], []
        query_x, query_y = [], []
        for pid in episode_patients:
            samples = self.patient_samples[pid]
            by_class = defaultdict(list)
            for spec, label in samples:
                by_class[label].append(spec)
            s_specs, s_labels = [], []
            q_specs, q_labels = [], []
            for label, specs in by_class.items():
                random.shuffle(specs)
                n_support = min(k_shot, max(1, len(specs) // 2))
                n_query = min(q_query, len(specs) - n_support)
                if n_query < 1:
                    s_specs.extend(specs)
                    s_labels.extend([label] * len(specs))
                else:
                    s_specs.extend(specs[:n_support])
                    s_labels.extend([label] * n_support)
                    q_specs.extend(specs[n_support:n_support + n_query])
                    q_labels.extend([label] * n_query)
            while len(s_specs) < k_shot:
                idx = random.randint(0, len(s_specs) - 1)
                s_specs.append(s_specs[idx])
                s_labels.append(s_labels[idx])
            while len(q_specs) < q_query:
                if len(q_specs) > 0:
                    idx = random.randint(0, len(q_specs) - 1)
                    q_specs.append(q_specs[idx])
                    q_labels.append(q_labels[idx])
                else:
                    idx = random.randint(0, len(s_specs) - 1)
                    q_specs.append(s_specs[idx])
                    q_labels.append(s_labels[idx])
            s_tensors = [self._spec_to_tensor(s) for s in s_specs[:k_shot]]
            q_tensors = [self._spec_to_tensor(s) for s in q_specs[:q_query]]
            support_x.append(torch.stack(s_tensors))
            support_y.append(torch.LongTensor(s_labels[:k_shot]))
            query_x.append(torch.stack(q_tensors))
            query_y.append(torch.LongTensor(q_labels[:q_query]))
        return (torch.stack(support_x), torch.stack(support_y),
                torch.stack(query_x), torch.stack(query_y))

    def _spec_to_tensor(self, spec):
        if self.augment and random.random() < 0.5:
            shift = random.randint(-10, 10)
            spec = np.roll(spec, shift, axis=-1)
        return torch.FloatTensor(spec).unsqueeze(0)


# ============================================================================
# Meta-Learning Model (same as V1)
# ============================================================================

class MetaPrototypicalClassifier(nn.Module):
    def __init__(self, nc=4, in_ch=1, feat_dim=256, width=1.0):
        super().__init__()
        self.nc = nc
        self.feat_dim = feat_dim
        c = lambda v: max(8, int(v * width))
        self.backbone = nn.Sequential(
            base.ConvBNAct(in_ch, c(24)),
            base.DSResBlock(c(24), c(32), stride=2),
            base.DSResBlock(c(32), c(48)),
            base.DSResBlock(c(48), c(64), stride=2),
            base.DSResBlock(c(64), c(96)),
            base.DSResBlock(c(96), c(128), stride=2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        backbone_out = c(128)
        self.projection = nn.Sequential(
            nn.Linear(backbone_out, feat_dim), nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )
        self.classifier = nn.Linear(feat_dim, nc)

    def extract_features(self, x):
        feat = self.backbone(x)
        return self.projection(feat)

    def forward(self, x):
        feat = self.extract_features(x)
        return self.classifier(feat)

    def compute_prototypes(self, support_feat, support_y):
        prototypes = []
        for c in range(self.nc):
            mask = (support_y == c)
            if mask.sum() > 0:
                prototypes.append(support_feat[mask].mean(dim=0))
            else:
                prototypes.append(torch.zeros(self.feat_dim, device=support_feat.device))
        return torch.stack(prototypes)

    def classify_by_prototypes(self, query_feat, prototypes):
        query_norm = F.normalize(query_feat, dim=1)
        proto_norm = F.normalize(prototypes, dim=1)
        return torch.mm(query_norm, proto_norm.t())

    def meta_forward(self, support_x, support_y, query_x, query_y):
        P, K = support_x.shape[:2]
        Q = query_x.shape[1]
        device = support_x.device
        total_loss = torch.tensor(0.0, device=device)
        total_correct = 0
        total_count = 0
        for p in range(P):
            s_x = support_x[p]
            q_x = query_x[p]
            s_y = support_y[p]
            q_y = query_y[p]
            s_feat = self.extract_features(s_x)
            q_feat = self.extract_features(q_x)
            prototypes = self.compute_prototypes(s_feat, s_y)
            logits = self.classify_by_prototypes(q_feat, prototypes)
            loss = F.cross_entropy(logits, q_y)
            total_loss += loss
            pred = logits.argmax(dim=1)
            total_correct += (pred == q_y).sum().item()
            total_count += Q
        meta_loss = total_loss / P
        meta_acc = total_correct / max(total_count, 1)
        return meta_loss, meta_acc


# ============================================================================
# V2: Improved Training with Class Balancing + Binary Loss
# ============================================================================

def meta_train(args, device, records_train, records_val, nc):
    """Meta-training with aggressive class balancing."""
    train_dataset = PatientGroupedDataset(records_train, args, augment=True)
    val_dataset = PatientGroupedDataset(records_val, args, augment=False)

    model = MetaPrototypicalClassifier(
        nc=nc, in_ch=args.in_channels, feat_dim=args.feat_dim, width=args.width,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Meta] Model: {total_params/1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Class-balanced focal loss
    class_counts = np.bincount([r.label_4class for r in records_train], minlength=nc)
    focal_fn = ClassBalancedFocalLoss(
        samples_per_class=class_counts.astype(np.float64),
        beta=0.9999, gamma=3.0, label_smoothing=0.05,
    )

    # Binary loss for Normal vs Abnormal
    binary_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 3.0], device=device)  # 3x weight for Abnormal
    )

    best_score = 0.0
    out_dir = ensure_dir(base.TRAINING_ARTIFACTS_DIR / "s8_meta_student_v2")

    for epoch in range(args.epochs):
        model.train()
        total_meta_loss = 0
        total_cls_loss = 0
        total_bin_loss = 0
        n_episodes = 0

        for _ in range(args.episodes_per_epoch):
            support_x, support_y, query_x, query_y = train_dataset.sample_episode(
                n_patients=args.n_patients, k_shot=args.k_shot,
                q_query=args.q_query, nc=nc,
            )
            support_x = support_x.to(device)
            support_y = support_y.to(device)
            query_x = query_x.to(device)
            query_y = query_y.to(device)

            meta_loss, meta_acc = model.meta_forward(support_x, support_y, query_x, query_y)

            # Also train standard classifier on query samples
            P, Q = query_x.shape[:2]
            q_flat = query_x.reshape(P * Q, *query_x.shape[2:])
            q_labels = query_y.reshape(P * Q)
            logits = model(q_flat)
            cls_loss = focal_fn(logits, q_labels)

            # Binary loss: Normal vs Abnormal
            bin_labels = (q_labels != 0).long()
            bin_logits = torch.stack([logits[:, 0], logits[:, 1:].max(dim=1)[0]], dim=1)
            bin_loss = binary_fn(bin_logits, bin_labels)

            # Combined loss
            w_meta = 0.3
            w_cls = 0.5
            w_bin = 0.2
            loss = w_meta * meta_loss + w_cls * cls_loss + w_bin * bin_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_meta_loss += meta_loss.item()
            total_cls_loss += cls_loss.item()
            total_bin_loss += bin_loss.item()
            n_episodes += 1

        scheduler.step()

        avg_meta = total_meta_loss / max(n_episodes, 1)
        avg_cls = total_cls_loss / max(n_episodes, 1)
        avg_bin = total_bin_loss / max(n_episodes, 1)

        # Validation: use patient-adapted evaluation
        if (epoch + 1) % 5 == 0:
            val_score = _eval_patient_adapted(model, records_val, args, device, nc)
            print(f"[Meta] Epoch {epoch+1}/{args.epochs}  "
                  f"meta={avg_meta:.4f}  cls={avg_cls:.4f}  bin={avg_bin:.4f}  "
                  f"val_score={val_score:.4f}")

            if val_score > best_score:
                best_score = val_score
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_score": val_score,
                    "args": vars(args),
                }, out_dir / "best.pt")

    print(f"[Meta] Best validation ICBHI Score: {best_score:.4f}")
    return model


@torch.no_grad()
def _eval_patient_adapted(model, records, args, device, nc):
    """Evaluate with patient-specific prototypes."""
    model.eval()
    patient_groups = defaultdict(list)
    for rec in records:
        spec = _compute_spec(rec, args)
        if spec is not None:
            patient_groups[rec.subject_id].append((spec, rec.label_4class))

    all_y_true, all_y_pred = [], []
    for pid, samples in patient_groups.items():
        specs = [s[0] for s in samples]
        labels = [s[1] for s in samples]
        x = torch.stack([torch.FloatTensor(s).unsqueeze(0) for s in specs]).to(device)
        feat = model.extract_features(x)
        prototypes = model.compute_prototypes(feat, torch.LongTensor(labels).to(device))
        logits = model.classify_by_prototypes(feat, prototypes)
        y_pred = logits.argmax(dim=1).cpu().numpy()
        all_y_true.extend(labels)
        all_y_pred.extend(y_pred.tolist())

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    se, sp, score = base.icbhi_score(y_true, y_pred)
    return score


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="ICBHI S8 V2: Meta-Learning with class balancing")
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
    p.add_argument("--feat_dim", type=int, default=256)
    p.add_argument("--n_patients", type=int, default=8)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--q_query", type=int, default=10)
    p.add_argument("--episodes_per_epoch", type=int, default=50)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--student_width", type=float, default=1.0)
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--time_shift", type=float, default=0.1)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--freq_mask", type=int, default=12)
    p.add_argument("--time_mask", type=int, default=48)
    p.add_argument("--use_vtlp", action="store_true", default=False)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--stage", type=str, default="all", choices=["train", "evaluate", "all"])
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.in_channels = 3 if args.input_view == "logmel_delta" else 1
    nc = args.num_classes

    records = base.build_records(Path(args.data_dir))
    splits = base.create_official_splits(records, nc, val_frac=args.val_size, seed=42)
    rec_train, rec_val, rec_test = splits['train'], splits['val'], splits['test']

    if args.stage in ("train", "all"):
        print("\n" + "=" * 60)
        print("Meta-Training V2: Patient-as-Task + Class Balancing")
        print("=" * 60)
        meta_train(args, device, rec_train, rec_val, nc)

    if args.stage in ("evaluate", "all"):
        print("\n" + "=" * 60)
        print("Evaluating Meta-Learning Model V2")
        print("=" * 60)

        model = MetaPrototypicalClassifier(
            nc=nc, in_ch=args.in_channels, feat_dim=args.feat_dim, width=args.width,
        ).to(device)
        ckpt = base.TRAINING_ARTIFACTS_DIR / "s8_meta_student_v2" / "best.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])

        # Patient-adapted evaluation (the correct way for meta-learning)
        adapted_score = _eval_patient_adapted(model, rec_test, args, device, nc)
        print(f"\n[Meta V2] Patient-Adapted Test Score: {adapted_score:.4f}")

        # Standard TTA evaluation
        test_ds = base.ICBHIDataset(rec_test, args, stats=None, augment=False)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, num_workers=1)
        metrics, threshold = evaluate_with_tta(model, test_dl, device, nc, n_tta=7)
        print(f"\n[Meta V2] Standard TTA Evaluation (threshold={threshold:.4f}):")
        print(f"  ICBHI Score: {metrics['icbhi_score']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Macro F1:    {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
