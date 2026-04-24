#!/usr/bin/env python3
"""
================================================================================
Layer 2: Random Forest Classifier + Cascaded Early-Exit Logic
================================================================================
Hệ thống chẩn đoán âm thanh hô hấp — Cascaded Framework (ICBHI 2017)

Chức năng:
  1. Đọc features.csv từ Layer 1 (ZCR, RMS, MFCCs 43-dim)
  2. Chia dữ liệu Subject-Independent (theo patient_id)
  3. Huấn luyện Random Forest tối ưu cho ARM embedded
  4. Cascaded Logic: Early-Exit nếu confidence > threshold, ngược lại → Layer 3 (CNN)
  5. Đánh giá: % Early-Exit vs % Pass-to-CNN → đo tiết kiệm tài nguyên

Output:
  - rf_model.joblib          — Mô hình RF serialized
  - rf_evaluation_report.txt — Báo cáo đánh giá chi tiết
  - cascaded_analysis.csv    — Phân tích từng sample: exit hay pass
  - threshold_sweep.png      — Biểu đồ phân tích ngưỡng tối ưu

Tối ưu ARM:
  - n_estimators tối ưu (50-200 cây)
  - max_depth giới hạn (10-20) → giảm memory
  - Inference < 5ms/sample trên ARM Cortex-A53

Usage:
    python layer2_random_forest.py
    python layer2_random_forest.py --threshold 0.75
    python layer2_random_forest.py --optimize   # Chạy grid search tối ưu

Author: Cascaded FPGA Framework
================================================================================
"""

import os
import sys
import time
import json
import argparse
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    import pickle
    HAS_JOBLIB = False

warnings.filterwarnings('ignore')

# ==============================================================================
# CẤU HÌNH (Configuration)
# ==============================================================================
# --- Paths ---
DEFAULT_FEATURES_CSV = '/home/iec/Parallel_Computing_on_FPGA/layer1_3/output/features.csv'
DEFAULT_LABELS_CSV = '/home/iec/Parallel_Computing_on_FPGA/data/combined/labels.csv'
DEFAULT_OUTPUT_DIR = '/home/iec/Parallel_Computing_on_FPGA/layer1_3/output_layer2'
DEFAULT_SPECTROGRAM_DIR = '/home/iec/Parallel_Computing_on_FPGA/layer1_3/output/spectrograms'

# --- Random Forest (tối ưu cho ARM Cortex-A53) ---
RF_N_ESTIMATORS = 100        # Số cây (cân bằng accuracy/speed)
RF_MAX_DEPTH = 15            # Giới hạn độ sâu → giảm memory
RF_MIN_SAMPLES_SPLIT = 5     # Tránh overfit trên dataset nhỏ
RF_MIN_SAMPLES_LEAF = 2      # Tránh lá quá nhỏ
RF_MAX_FEATURES = 'sqrt'     # Feature subsampling → giảm correlation
RF_N_JOBS = -1               # Dùng tất cả CPU khi train
RF_RANDOM_STATE = 42

# --- Cascaded Threshold ---
DEFAULT_THRESHOLD = 0.4     # Ngưỡng confidence mặc định
THRESHOLD_SWEEP_RANGE = np.arange(0.40, 0.96, 0.05)  # Dải quét ngưỡng

# --- Cross-Validation ---
N_FOLDS = 5                  # Subject-Independent K-Fold

# --- Classes ---
CLASS_NAMES = ['COPD', 'Healthy', 'Non-COPD']


# ==============================================================================
# MODULE 1: DATA LOADER
# ==============================================================================
class DataLoader:
    """Nạp và chuẩn bị dữ liệu từ features.csv + labels.csv."""

    def __init__(self, features_csv: str, labels_csv: str):
        self.features_csv = features_csv
        self.labels_csv = labels_csv
        self._filenames = None

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Load features + labels + patient_ids.

        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Label vector (n_samples,) — encoded integers
            patient_ids: Patient ID array for GroupKFold
            feature_names: Tên các cột đặc trưng
            class_names: Tên các nhãn
        """
        # Load features
        df_features = pd.read_csv(self.features_csv)
        print(f"  📊 Features loaded: {df_features.shape[0]} samples × "
              f"{df_features.shape[1]} columns")

        # Load labels để lấy patient_id
        df_labels = pd.read_csv(self.labels_csv)

        # Merge patient_id vào features
        df_merged = df_features.merge(
            df_labels[['filename', 'patient_id']],
            on='filename', how='left'
        )

        # Xử lý trường hợp không tìm được patient_id
        missing_mask = df_merged['patient_id'].isna()
        if missing_mask.any():
            print(f"  ⚠️  {missing_mask.sum()} samples thiếu patient_id → extract từ filename")
            for idx in df_merged[missing_mask].index:
                fname = df_merged.loc[idx, 'filename']
                parts = fname.split('_')
                if len(parts) >= 2:
                    df_merged.loc[idx, 'patient_id'] = f"{parts[0]}_{parts[1]}"
                else:
                    df_merged.loc[idx, 'patient_id'] = f"unknown_{idx}"

        # Lưu filenames và labels cho tra cứu spectrogram path
        self._filenames = df_merged['filename'].values.tolist()
        self._labels = df_merged['label'].values.tolist()

        # Tách features, labels, patient_ids
        feature_cols = [c for c in df_features.columns if c not in ['filename', 'label']]
        X = df_merged[feature_cols].values.astype(np.float32)
        feature_names = feature_cols

        # Encode labels
        le = LabelEncoder()
        le.fit(CLASS_NAMES)  # Đảm bảo thứ tự cố định
        y = le.transform(df_merged['label'].values)
        class_names = list(le.classes_)

        patient_ids = df_merged['patient_id'].values

        # Xử lý NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"  📐 Feature matrix: {X.shape}")
        print(f"  🏷️  Classes: {class_names}")
        print(f"  👥 Unique patients: {len(np.unique(patient_ids))}")

        # Class distribution
        for i, cls in enumerate(class_names):
            count = np.sum(y == i)
            print(f"     {cls}: {count} samples ({count/len(y)*100:.1f}%)")

        return X, y, patient_ids, feature_names, class_names

    def get_filenames(self) -> List[str]:
        """Trả về danh sách filename (sau khi đã load)."""
        return self._filenames or []

    def get_labels(self) -> List[str]:
        """Trả về danh sách label string (sau khi đã load)."""
        return self._labels or []


# ==============================================================================
# MODULE 2: RANDOM FOREST TRAINER
# ==============================================================================
class RandomForestTrainer:
    """
    Huấn luyện và đánh giá Random Forest với Subject-Independent splitting.

    Tối ưu cho ARM:
      - n_estimators=100 (< 5ms inference trên Cortex-A53)
      - max_depth=15 (giảm memory footprint)
      - Xuất model nhẹ cho embedded deployment
    """

    def __init__(self, n_estimators: int = RF_N_ESTIMATORS,
                 max_depth: int = RF_MAX_DEPTH,
                 random_state: int = RF_RANDOM_STATE):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()

    def create_model(self) -> RandomForestClassifier:
        """Khởi tạo RF model với hyper-parameters tối ưu ARM."""
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features=RF_MAX_FEATURES,
            class_weight='balanced',       # Xử lý imbalanced dataset
            n_jobs=RF_N_JOBS,
            random_state=self.random_state,
            oob_score=True,                # Out-of-Bag score
        )

    def train_and_evaluate_cv(self, X: np.ndarray, y: np.ndarray,
                               patient_ids: np.ndarray,
                               class_names: List[str]) -> Dict:
        """
        Subject-Independent Cross-Validation.

        Chia theo patient_id → không có data leakage giữa train/test.
        """
        print(f"\n{'='*70}")
        print(f"  TRAINING: Random Forest ({self.n_estimators} trees, "
              f"max_depth={self.max_depth})")
        print(f"  Cross-Validation: {N_FOLDS}-Fold Subject-Independent")
        print(f"{'='*70}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # GroupKFold theo patient_id
        gkf = GroupKFold(n_splits=N_FOLDS)

        fold_results = []
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        all_fold_ids = []

        for fold_id, (train_idx, test_idx) in enumerate(
            gkf.split(X_scaled, y, groups=patient_ids)
        ):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train
            model = self.create_model()
            t_start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t_start

            # Predict
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            fold_results.append({
                'fold': fold_id + 1,
                'accuracy': acc,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'train_time': train_time,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'oob_score': model.oob_score_ if hasattr(model, 'oob_score_') else 0,
            })

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_proba.extend(y_proba)
            all_fold_ids.extend([fold_id] * len(test_idx))

            print(f"  Fold {fold_id+1}/{N_FOLDS} | "
                  f"Acc: {acc*100:.2f}% | "
                  f"Macro F1: {f1_macro*100:.2f}% | "
                  f"Weighted F1: {f1_weighted*100:.2f}% | "
                  f"Train: {train_time:.2f}s | "
                  f"OOB: {model.oob_score_*100:.2f}%")

        # Aggregate results
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_proba = np.array(all_y_proba)

        mean_acc = np.mean([r['accuracy'] for r in fold_results])
        std_acc = np.std([r['accuracy'] for r in fold_results])
        mean_f1 = np.mean([r['f1_macro'] for r in fold_results])
        std_f1 = np.std([r['f1_macro'] for r in fold_results])

        print(f"\n{'='*70}")
        print(f"  {N_FOLDS}-FOLD CV RESULTS (Subject-Independent)")
        print(f"{'='*70}")
        print(f"  Accuracy:    {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print(f"  Macro F1:    {mean_f1*100:.2f}% ± {std_f1*100:.2f}%")
        print(f"{'='*70}")

        # Overall classification report
        print(f"\n  Classification Report (Aggregated across all folds):")
        print(classification_report(
            all_y_true, all_y_pred,
            target_names=class_names, zero_division=0
        ))

        # Confusion Matrix
        cm = confusion_matrix(all_y_true, all_y_pred)
        print(f"  Confusion Matrix:")
        print(f"  {'':15s}  " + "  ".join(f"{c:>10s}" for c in class_names))
        for i, row in enumerate(cm):
            print(f"  {class_names[i]:15s}  " +
                  "  ".join(f"{v:10d}" for v in row))

        return {
            'fold_results': fold_results,
            'y_true': all_y_true,
            'y_pred': all_y_pred,
            'y_proba': all_y_proba,
            'fold_ids': all_fold_ids,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'mean_f1_macro': mean_f1,
            'std_f1_macro': std_f1,
            'confusion_matrix': cm,
        }

    def train_final_model(self, X: np.ndarray, y: np.ndarray):
        """Train final model trên toàn bộ dataset."""
        print(f"\n  🎯 Training final model on ALL data ({len(X)} samples)...")
        X_scaled = self.scaler.fit_transform(X)
        self.model = self.create_model()
        self.model.fit(X_scaled, y)
        print(f"  ✅ Final model trained. OOB Score: {self.model.oob_score_*100:.2f}%")

    def save_model(self, output_dir: str):
        """Lưu model + scaler."""
        output_dir = Path(output_dir)
        if HAS_JOBLIB:
            joblib.dump(self.model, output_dir / 'rf_model.joblib')
            joblib.dump(self.scaler, output_dir / 'rf_scaler.joblib')
            print(f"  💾 Model saved: {output_dir / 'rf_model.joblib'}")
        else:
            with open(output_dir / 'rf_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            with open(output_dir / 'rf_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"  💾 Model saved: {output_dir / 'rf_model.pkl'}")

    def get_feature_importance(self, feature_names: List[str],
                               top_k: int = 15) -> pd.DataFrame:
        """Trả về top-K features quan trọng nhất."""
        if self.model is None:
            return pd.DataFrame()
        importance = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return df.head(top_k)


# ==============================================================================
# MODULE 3: CASCADED LOGIC — EARLY EXIT / PASS TO LAYER 3
# ==============================================================================
class CascadedPredictor:
    """
    Cơ chế phân tầng (Cascaded Early-Exit) cho Layer 2.

    Logic:
      - RF predict + probability
      - Nếu max_probability >= threshold → EARLY EXIT (kết quả từ RF)
      - Nếu max_probability < threshold  → UNCERTAIN → chuyển sang Layer 3 (CNN)

    Tối ưu:
      - Mẫu "dễ" (high confidence) xử lý bởi RF nhẹ → tiết kiệm năng lượng
      - Mẫu "khó" (low confidence) mới cần CNN nặng
    """

    def __init__(self, model: RandomForestClassifier,
                 scaler: StandardScaler,
                 class_names: List[str],
                 threshold: float = DEFAULT_THRESHOLD,
                 spectrogram_dir: str = DEFAULT_SPECTROGRAM_DIR):
        self.model = model
        self.scaler = scaler
        self.class_names = class_names
        self.threshold = threshold
        self.spectrogram_dir = Path(spectrogram_dir)

    def _get_spectrogram_path(self, filename: str, label: str) -> str:
        """
        Tìm đường dẫn ảnh Hybrid Spectrogram tương ứng.
        Ảnh được lưu bởi Layer 1 tại: spectrograms/{label}/{stem}.png
        """
        stem = Path(filename).stem
        spec_path = self.spectrogram_dir / label / f"{stem}.png"
        if spec_path.exists():
            return str(spec_path)
        # Fallback: tìm trong tất cả thư mục class
        for cls in self.class_names:
            alt_path = self.spectrogram_dir / cls / f"{stem}.png"
            if alt_path.exists():
                return str(alt_path)
        return str(spec_path)  # Trả về path mặc định dù chưa tồn tại

    def predict_with_threshold(self, sample: np.ndarray,
                                threshold: Optional[float] = None,
                                filename: Optional[str] = None,
                                label: Optional[str] = None) -> Dict:
        """
        Dự đoán với cơ chế phân tầng.

        Args:
            sample: Feature vector (1, n_features) hoặc (n_features,)
            threshold: Ngưỡng confidence (mặc định dùng self.threshold)
            filename: Tên file .wav (để tra cứu spectrogram path)
            label: Label string (để xác định thư mục spectrogram)

        Returns:
            dict:
                prediction: Tên class dự đoán hoặc 'Uncertain: Pass to Layer 3'
                confidence: Xác suất cao nhất
                action: 'EARLY_EXIT' hoặc 'PASS_TO_LAYER3'
                probabilities: Xác suất từng class
                spectrogram_path: (chỉ khi PASS_TO_LAYER3) đường dẫn ảnh cho DPU
        """
        if threshold is None:
            threshold = self.threshold

        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        sample_scaled = self.scaler.transform(sample)

        proba = self.model.predict_proba(sample_scaled)[0]
        max_prob = float(np.max(proba))
        pred_idx = int(np.argmax(proba))
        pred_class = self.class_names[pred_idx]

        if max_prob >= threshold:
            return {
                'prediction': pred_class,
                'confidence': max_prob,
                'action': 'EARLY_EXIT',
                'probabilities': {
                    self.class_names[i]: float(proba[i])
                    for i in range(len(self.class_names))
                }
            }
        else:
            result = {
                'prediction': 'Uncertain: Pass to Layer 3',
                'confidence': max_prob,
                'action': 'PASS_TO_LAYER3',
                'probabilities': {
                    self.class_names[i]: float(proba[i])
                    for i in range(len(self.class_names))
                },
                'rf_suggestion': pred_class,
            }
            # Đính kèm đường dẫn Hybrid Spectrogram cho DPU/CNN
            if filename and label:
                result['spectrogram_path'] = self._get_spectrogram_path(filename, label)
            return result

    def analyze_cascaded_performance(self, X: np.ndarray, y: np.ndarray,
                                      y_proba: np.ndarray,
                                      threshold: Optional[float] = None) -> Dict:
        """
        Phân tích hiệu suất cascaded trên toàn bộ dataset.

        Đo:
          - % Early Exit (RF xử lý)
          - % Pass to Layer 3 (CNN xử lý)
          - Accuracy trên phần Early Exit
          - Accuracy trên phần Uncertain
        """
        if threshold is None:
            threshold = self.threshold

        max_proba = np.max(y_proba, axis=1)
        y_pred_rf = np.argmax(y_proba, axis=1)

        # Phân loại: confident (exit) vs uncertain (pass)
        confident_mask = max_proba >= threshold
        uncertain_mask = ~confident_mask

        n_total = len(y)
        n_exit = np.sum(confident_mask)
        n_pass = np.sum(uncertain_mask)

        result = {
            'threshold': threshold,
            'total_samples': n_total,
            'early_exit_count': int(n_exit),
            'early_exit_pct': float(n_exit / n_total * 100),
            'pass_to_cnn_count': int(n_pass),
            'pass_to_cnn_pct': float(n_pass / n_total * 100),
        }

        # Accuracy trên phần Early Exit
        if n_exit > 0:
            exit_acc = accuracy_score(y[confident_mask], y_pred_rf[confident_mask])
            exit_f1 = f1_score(y[confident_mask], y_pred_rf[confident_mask],
                              average='macro', zero_division=0)
            result['early_exit_accuracy'] = float(exit_acc)
            result['early_exit_f1'] = float(exit_f1)
        else:
            result['early_exit_accuracy'] = 0.0
            result['early_exit_f1'] = 0.0

        # Accuracy trên phần Uncertain (RF prediction, dù sẽ chuyển CNN)
        if n_pass > 0:
            pass_acc = accuracy_score(y[uncertain_mask], y_pred_rf[uncertain_mask])
            result['uncertain_rf_accuracy'] = float(pass_acc)
        else:
            result['uncertain_rf_accuracy'] = 0.0

        # Per-class analysis
        result['per_class'] = {}
        for i, cls in enumerate(self.class_names):
            cls_mask = y == i
            if cls_mask.sum() == 0:
                continue
            cls_confident = confident_mask & cls_mask
            cls_total = cls_mask.sum()
            cls_exit = cls_confident.sum()
            result['per_class'][cls] = {
                'total': int(cls_total),
                'early_exit': int(cls_exit),
                'exit_pct': float(cls_exit / cls_total * 100),
                'pass_pct': float((cls_total - cls_exit) / cls_total * 100),
            }

        return result


# ==============================================================================
# MODULE 4: THRESHOLD OPTIMIZER
# ==============================================================================
class ThresholdOptimizer:
    """
    Quét và tìm ngưỡng tối ưu cho cascaded early-exit.

    Trade-off:
      - Threshold cao → ít early-exit, accuracy cao hơn trên phần exit
      - Threshold thấp → nhiều early-exit, tiết kiệm tài nguyên nhưng accuracy giảm
    """

    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    def sweep_thresholds(self, y_true: np.ndarray, y_proba: np.ndarray,
                          thresholds: np.ndarray = THRESHOLD_SWEEP_RANGE) -> List[Dict]:
        """Quét các ngưỡng và trả về metrics cho từng ngưỡng."""
        results = []
        max_proba = np.max(y_proba, axis=1)
        y_pred = np.argmax(y_proba, axis=1)

        for thresh in thresholds:
            confident = max_proba >= thresh
            n_exit = confident.sum()
            n_total = len(y_true)

            if n_exit > 0:
                exit_acc = accuracy_score(y_true[confident], y_pred[confident])
                exit_f1 = f1_score(y_true[confident], y_pred[confident],
                                  average='macro', zero_division=0)
            else:
                exit_acc = 0.0
                exit_f1 = 0.0

            results.append({
                'threshold': float(thresh),
                'exit_pct': float(n_exit / n_total * 100),
                'pass_pct': float((n_total - n_exit) / n_total * 100),
                'exit_accuracy': float(exit_acc),
                'exit_f1_macro': float(exit_f1),
                'exit_count': int(n_exit),
                'pass_count': int(n_total - n_exit),
            })

        return results

    def find_optimal_threshold(self, sweep_results: List[Dict],
                                min_accuracy: float = 0.85,
                                min_exit_pct: float = 50.0) -> float:
        """
        Tìm ngưỡng tối ưu: maximize exit_pct trong khi exit_accuracy >= min_accuracy.

        Nếu không tìm được, trả về ngưỡng cho accuracy cao nhất.
        """
        candidates = [
            r for r in sweep_results
            if r['exit_accuracy'] >= min_accuracy and r['exit_pct'] >= min_exit_pct
        ]

        if candidates:
            # Maximize exit_pct (tiết kiệm tài nguyên nhất)
            best = max(candidates, key=lambda r: r['exit_pct'])
            return best['threshold']
        else:
            # Fallback: chọn accuracy cao nhất
            best = max(sweep_results, key=lambda r: r['exit_accuracy'])
            return best['threshold']

    def plot_threshold_analysis(self, sweep_results: List[Dict],
                                 optimal_threshold: float,
                                 save_path: str):
        """Vẽ biểu đồ phân tích ngưỡng."""
        thresholds = [r['threshold'] for r in sweep_results]
        exit_pcts = [r['exit_pct'] for r in sweep_results]
        exit_accs = [r['exit_accuracy'] * 100 for r in sweep_results]
        exit_f1s = [r['exit_f1_macro'] * 100 for r in sweep_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Exit % vs Threshold
        ax1.plot(thresholds, exit_pcts, 'b-o', lw=2, markersize=5, label='Early Exit %')
        pass_pcts = [100 - e for e in exit_pcts]
        ax1.plot(thresholds, pass_pcts, 'r-s', lw=2, markersize=5, label='Pass to CNN %')
        ax1.axvline(x=optimal_threshold, color='green', linestyle='--', lw=2,
                    label=f'Optimal τ={optimal_threshold:.2f}')
        ax1.set_xlabel('Confidence Threshold (τ)', fontsize=12)
        ax1.set_ylabel('Percentage (%)', fontsize=12)
        ax1.set_title('Cascaded Exit Distribution', fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)

        # Plot 2: Accuracy/F1 vs Threshold
        ax2.plot(thresholds, exit_accs, 'g-o', lw=2, markersize=5, label='Exit Accuracy')
        ax2.plot(thresholds, exit_f1s, 'm-^', lw=2, markersize=5, label='Exit Macro F1')
        ax2.axvline(x=optimal_threshold, color='green', linestyle='--', lw=2,
                    label=f'Optimal τ={optimal_threshold:.2f}')
        ax2.set_xlabel('Confidence Threshold (τ)', fontsize=12)
        ax2.set_ylabel('Score (%)', fontsize=12)
        ax2.set_title('Early Exit Quality', fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📊 Threshold analysis plot saved: {save_path}")


# ==============================================================================
# MODULE 5: REPORT GENERATOR
# ==============================================================================
class ReportGenerator:
    """Tạo báo cáo đánh giá chi tiết."""

    @staticmethod
    def generate_report(cv_results: Dict, cascaded_results: Dict,
                         sweep_results: List[Dict], optimal_threshold: float,
                         feature_importance: pd.DataFrame,
                         class_names: List[str],
                         output_path: str):
        """Tạo báo cáo text đầy đủ."""
        lines = []
        lines.append("=" * 70)
        lines.append("  LAYER 2: RANDOM FOREST EVALUATION REPORT")
        lines.append("=" * 70)

        # --- CV Results ---
        lines.append("\n[1] CROSS-VALIDATION RESULTS (Subject-Independent)")
        lines.append("-" * 50)
        lines.append(f"  Accuracy:    {cv_results['mean_accuracy']*100:.2f}% "
                     f"± {cv_results['std_accuracy']*100:.2f}%")
        lines.append(f"  Macro F1:    {cv_results['mean_f1_macro']*100:.2f}% "
                     f"± {cv_results['std_f1_macro']*100:.2f}%")

        lines.append("\n  Per-Fold Results:")
        for r in cv_results['fold_results']:
            lines.append(f"    Fold {r['fold']} | "
                        f"Acc: {r['accuracy']*100:.2f}% | "
                        f"F1: {r['f1_macro']*100:.2f}% | "
                        f"OOB: {r['oob_score']*100:.2f}%")

        # --- Confusion Matrix ---
        lines.append(f"\n  Confusion Matrix:")
        cm = cv_results['confusion_matrix']
        lines.append(f"  {'':15s}  " + "  ".join(f"{c:>10s}" for c in class_names))
        for i, row in enumerate(cm):
            lines.append(f"  {class_names[i]:15s}  " +
                        "  ".join(f"{v:10d}" for v in row))

        # --- Cascaded Analysis ---
        lines.append(f"\n\n[2] CASCADED EARLY-EXIT ANALYSIS (τ = {cascaded_results['threshold']:.2f})")
        lines.append("-" * 50)
        lines.append(f"  Total samples:       {cascaded_results['total_samples']}")
        lines.append(f"  Early Exit (RF):     {cascaded_results['early_exit_count']} "
                     f"({cascaded_results['early_exit_pct']:.1f}%)")
        lines.append(f"  Pass to CNN:         {cascaded_results['pass_to_cnn_count']} "
                     f"({cascaded_results['pass_to_cnn_pct']:.1f}%)")
        lines.append(f"  Exit Accuracy:       {cascaded_results['early_exit_accuracy']*100:.2f}%")
        lines.append(f"  Exit F1 (Macro):     {cascaded_results['early_exit_f1']*100:.2f}%")

        lines.append(f"\n  Per-Class Breakdown:")
        for cls, data in cascaded_results.get('per_class', {}).items():
            lines.append(f"    {cls:15s} | "
                        f"Exit: {data['early_exit']:4d}/{data['total']:4d} "
                        f"({data['exit_pct']:.1f}%) | "
                        f"Pass: {data['pass_pct']:.1f}%")

        # --- Resource Savings ---
        lines.append(f"\n\n[3] RESOURCE SAVINGS ESTIMATE")
        lines.append("-" * 50)
        exit_pct = cascaded_results['early_exit_pct']
        lines.append(f"  RF xử lý:                   {exit_pct:.1f}% samples")
        lines.append(f"  CNN cần xử lý:              {100-exit_pct:.1f}% samples")
        lines.append(f"  Tiết kiệm CNN inference:    ~{exit_pct:.0f}% computation")
        lines.append(f"  RF inference time (ARM):     < 5ms/sample (estimated)")
        lines.append(f"  CNN inference time (FPGA):   ~50ms/sample (estimated)")
        energy_saving = exit_pct * 0.9  # RF dùng ~10% năng lượng so với CNN
        lines.append(f"  Ước tính tiết kiệm NL:      ~{energy_saving:.0f}%")

        # --- Threshold Sweep ---
        lines.append(f"\n\n[4] THRESHOLD SWEEP ANALYSIS")
        lines.append("-" * 50)
        lines.append(f"  Optimal threshold: τ = {optimal_threshold:.2f}")
        lines.append(f"\n  {'Threshold':>10s}  {'Exit%':>8s}  {'Pass%':>8s}  "
                     f"{'ExitAcc':>8s}  {'ExitF1':>8s}")
        lines.append(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        for r in sweep_results:
            marker = " ◀" if abs(r['threshold'] - optimal_threshold) < 0.01 else ""
            lines.append(f"  {r['threshold']:10.2f}  "
                        f"{r['exit_pct']:7.1f}%  "
                        f"{r['pass_pct']:7.1f}%  "
                        f"{r['exit_accuracy']*100:7.2f}%  "
                        f"{r['exit_f1_macro']*100:7.2f}%{marker}")

        # --- Feature Importance ---
        lines.append(f"\n\n[5] TOP FEATURE IMPORTANCE")
        lines.append("-" * 50)
        for _, row in feature_importance.iterrows():
            bar = "█" * int(row['importance'] * 100)
            lines.append(f"  {row['feature']:25s}  {row['importance']:.4f}  {bar}")

        lines.append(f"\n{'='*70}")
        lines.append(f"  END OF REPORT")
        lines.append(f"{'='*70}")

        report_text = "\n".join(lines)

        with open(output_path, 'w') as f:
            f.write(report_text)

        print(report_text)
        print(f"\n  💾 Report saved: {output_path}")

    @staticmethod
    def save_cascaded_csv(y_true: np.ndarray, y_pred: np.ndarray,
                           y_proba: np.ndarray, class_names: List[str],
                           threshold: float, output_path: str,
                           filenames: Optional[List[str]] = None,
                           labels: Optional[List[str]] = None,
                           spectrogram_dir: Optional[str] = None):
        """Lưu phân tích chi tiết từng sample vào CSV (kèm spectrogram path)."""
        rows = []
        for i in range(len(y_true)):
            max_prob = float(np.max(y_proba[i]))
            pred_class = class_names[int(np.argmax(y_proba[i]))]
            true_class = class_names[int(y_true[i])]
            action = 'EARLY_EXIT' if max_prob >= threshold else 'PASS_TO_LAYER3'
            correct = pred_class == true_class

            row = {
                'sample_idx': i,
                'filename': filenames[i] if filenames else '',
                'true_label': true_class,
                'rf_prediction': pred_class,
                'confidence': round(max_prob, 4),
                'action': action,
                'correct': correct,
                **{f'prob_{cls}': round(float(y_proba[i][j]), 4)
                   for j, cls in enumerate(class_names)}
            }

            # Đính kèm spectrogram path cho mẫu uncertain
            if action == 'PASS_TO_LAYER3' and filenames and labels and spectrogram_dir:
                stem = Path(filenames[i]).stem
                label = labels[i] if i < len(labels) else true_class
                row['spectrogram_path'] = str(
                    Path(spectrogram_dir) / label / f"{stem}.png"
                )
            else:
                row['spectrogram_path'] = ''

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"  💾 Cascaded analysis saved: {output_path} ({len(rows)} rows)")

        # Xuất danh sách mẫu uncertain riêng cho DPU
        uncertain_df = df[df['action'] == 'PASS_TO_LAYER3']
        if len(uncertain_df) > 0:
            dpu_path = str(Path(output_path).parent / 'uncertain_for_dpu.csv')
            uncertain_df[['filename', 'true_label', 'rf_prediction',
                          'confidence', 'spectrogram_path']].to_csv(
                dpu_path, index=False
            )
            print(f"  🔀 Uncertain samples for DPU: {dpu_path} "
                  f"({len(uncertain_df)} samples)")


# ==============================================================================
# MODULE 6: MODEL OPTIMIZER (Grid Search cho ARM)
# ==============================================================================
class ARMOptimizer:
    """Tìm hyperparameters tối ưu cho ARM Cortex-A53."""

    PARAM_GRID = {
        'n_estimators': [50, 75, 100, 150, 200],
        'max_depth': [8, 10, 12, 15, 20],
    }

    @staticmethod
    def optimize(X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray,
                  class_names: List[str]) -> Dict:
        """Grid search với Subject-Independent CV."""
        print(f"\n{'='*70}")
        print(f"  ARM OPTIMIZATION: Grid Search")
        print(f"{'='*70}")

        best_score = -1.0
        best_params = {}
        results = []

        gkf = GroupKFold(n_splits=3)  # 3-fold cho tốc độ
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for n_est in ARMOptimizer.PARAM_GRID['n_estimators']:
            for depth in ARMOptimizer.PARAM_GRID['max_depth']:
                fold_scores = []
                t_inference_total = 0

                for train_idx, test_idx in gkf.split(X_scaled, y, groups=patient_ids):
                    rf = RandomForestClassifier(
                        n_estimators=n_est, max_depth=depth,
                        min_samples_split=5, min_samples_leaf=2,
                        max_features='sqrt', class_weight='balanced',
                        random_state=42, n_jobs=-1
                    )
                    rf.fit(X_scaled[train_idx], y[train_idx])

                    t0 = time.time()
                    y_pred = rf.predict(X_scaled[test_idx])
                    t_inference = (time.time() - t0) / len(test_idx) * 1000  # ms/sample
                    t_inference_total += t_inference

                    f1 = f1_score(y[test_idx], y_pred, average='macro', zero_division=0)
                    fold_scores.append(f1)

                mean_f1 = np.mean(fold_scores)
                avg_inference = t_inference_total / 3

                results.append({
                    'n_estimators': n_est,
                    'max_depth': depth,
                    'f1_macro': mean_f1,
                    'inference_ms': avg_inference,
                })

                marker = ""
                if mean_f1 > best_score:
                    best_score = mean_f1
                    best_params = {'n_estimators': n_est, 'max_depth': depth}
                    marker = " ★ BEST"

                print(f"  n_est={n_est:3d}, depth={depth:2d} → "
                      f"F1: {mean_f1*100:.2f}%, "
                      f"Inference: {avg_inference:.2f}ms/sample{marker}")

        print(f"\n  🏆 Best: n_estimators={best_params['n_estimators']}, "
              f"max_depth={best_params['max_depth']} → F1: {best_score*100:.2f}%")

        return best_params


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Layer 2: Random Forest + Cascaded Early-Exit"
    )
    parser.add_argument('--features_csv', type=str, default=DEFAULT_FEATURES_CSV,
                        help='Path to features.csv from Layer 1')
    parser.add_argument('--labels_csv', type=str, default=DEFAULT_LABELS_CSV,
                        help='Path to labels.csv with patient_id')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory')
    parser.add_argument('--spectrogram_dir', type=str, default=DEFAULT_SPECTROGRAM_DIR,
                        help='Path to spectrograms/ from Layer 1 (for DPU linking)')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help='Confidence threshold cho early-exit')
    parser.add_argument('--optimize', action='store_true',
                        help='Chạy grid search tối ưu hyperparameters')
    parser.add_argument('--n_estimators', type=int, default=RF_N_ESTIMATORS,
                        help='Số cây Random Forest')
    parser.add_argument('--max_depth', type=int, default=RF_MAX_DEPTH,
                        help='Độ sâu tối đa')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  LAYER 2: Random Forest + Cascaded Early-Exit Pipeline")
    print("=" * 70)
    print(f"  Spectrogram Dir: {args.spectrogram_dir}")

    # ============================
    # Step 1: Load Data
    # ============================
    print("\n[1/5] 📂 Loading data...")
    loader = DataLoader(args.features_csv, args.labels_csv)
    X, y, patient_ids, feature_names, class_names = loader.load()
    filenames = loader.get_filenames()
    label_strings = loader.get_labels()

    # ============================
    # Step 2: Optimize (optional)
    # ============================
    n_est = args.n_estimators
    max_depth = args.max_depth

    if args.optimize:
        best_params = ARMOptimizer.optimize(X, y, patient_ids, class_names)
        n_est = best_params['n_estimators']
        max_depth = best_params['max_depth']

    # ============================
    # Step 3: Train & Evaluate (CV)
    # ============================
    print(f"\n[2/5] 🌲 Training Random Forest...")
    trainer = RandomForestTrainer(n_estimators=n_est, max_depth=max_depth)
    cv_results = trainer.train_and_evaluate_cv(X, y, patient_ids, class_names)

    # Train final model
    trainer.train_final_model(X, y)
    trainer.save_model(str(output_dir))

    # Feature importance
    feat_imp = trainer.get_feature_importance(feature_names, top_k=15)

    # ============================
    # Step 4: Cascaded Analysis
    # ============================
    print(f"\n[3/5] 🔀 Cascaded Early-Exit Analysis...")

    predictor = CascadedPredictor(
        trainer.model, trainer.scaler, class_names, args.threshold,
        spectrogram_dir=args.spectrogram_dir
    )

    # Analyze với cross-validation probabilities
    cascaded = predictor.analyze_cascaded_performance(
        X, cv_results['y_true'], cv_results['y_proba'], args.threshold
    )

    print(f"\n  Threshold τ = {args.threshold:.2f}")
    print(f"  ├── Early Exit (RF):  {cascaded['early_exit_count']} "
          f"({cascaded['early_exit_pct']:.1f}%)")
    print(f"  ├── Pass to CNN:      {cascaded['pass_to_cnn_count']} "
          f"({cascaded['pass_to_cnn_pct']:.1f}%)")
    print(f"  ├── Exit Accuracy:    {cascaded['early_exit_accuracy']*100:.2f}%")
    print(f"  └── Exit F1 (Macro):  {cascaded['early_exit_f1']*100:.2f}%")

    # ============================
    # Step 5: Threshold Sweep
    # ============================
    print(f"\n[4/5] 📈 Threshold Sweep Analysis...")
    optimizer = ThresholdOptimizer(class_names)
    sweep_results = optimizer.sweep_thresholds(
        cv_results['y_true'], cv_results['y_proba']
    )
    optimal_threshold = optimizer.find_optimal_threshold(sweep_results)
    print(f"  🎯 Optimal threshold: τ = {optimal_threshold:.2f}")

    optimizer.plot_threshold_analysis(
        sweep_results, optimal_threshold,
        str(output_dir / 'threshold_sweep.png')
    )

    # Re-analyze with optimal threshold
    cascaded_optimal = predictor.analyze_cascaded_performance(
        X, cv_results['y_true'], cv_results['y_proba'], optimal_threshold
    )

    # ============================
    # Step 6: Generate Reports + DPU Export
    # ============================
    print(f"\n[5/5] 📋 Generating Reports + DPU Uncertain List...")

    ReportGenerator.generate_report(
        cv_results, cascaded_optimal, sweep_results,
        optimal_threshold, feat_imp, class_names,
        str(output_dir / 'rf_evaluation_report.txt')
    )

    ReportGenerator.save_cascaded_csv(
        cv_results['y_true'], cv_results['y_pred'],
        cv_results['y_proba'], class_names,
        optimal_threshold,
        str(output_dir / 'cascaded_analysis.csv'),
        filenames=filenames,
        labels=label_strings,
        spectrogram_dir=args.spectrogram_dir,
    )

    # Save config
    config = {
        'n_estimators': n_est,
        'max_depth': max_depth,
        'optimal_threshold': float(optimal_threshold),
        'user_threshold': float(args.threshold),
        'cv_accuracy': float(cv_results['mean_accuracy']),
        'cv_f1_macro': float(cv_results['mean_f1_macro']),
        'early_exit_pct': float(cascaded_optimal['early_exit_pct']),
        'exit_accuracy': float(cascaded_optimal['early_exit_accuracy']),
        'class_names': class_names,
        'feature_names': feature_names,
        'spectrogram_dir': args.spectrogram_dir,
    }
    with open(output_dir / 'layer2_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  ✅ LAYER 2 PIPELINE HOÀN TẤT")
    print(f"{'='*70}")
    print(f"  📁 Output: {output_dir}/")
    print(f"     ├── rf_model.joblib           — Mô hình RF")
    print(f"     ├── rf_scaler.joblib          — Feature scaler")
    print(f"     ├── rf_evaluation_report.txt  — Báo cáo đánh giá")
    print(f"     ├── cascaded_analysis.csv     — Chi tiết từng sample")
    print(f"     ├── uncertain_for_dpu.csv     — Mẫu uncertain + spectrogram paths")
    print(f"     ├── threshold_sweep.png       — Biểu đồ ngưỡng")
    print(f"     └── layer2_config.json        — Cấu hình")
    print(f"  🔀 Layer 3 ready: Mẫu uncertain có sẵn spectrogram path cho DPU")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
