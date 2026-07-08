"""
Stacking Meta-Learner for ICBHI Ensemble

Instead of fixed weighted average, train a logistic regression meta-learner
on concatenated logits from all models. This captures model interactions
(e.g., "when BEATs says wheeze and CLAP says crackle, the answer is usually both").

Usage:
    python scripts/stacking_ensemble.py \
        --logits_dirs teacher_logits teacher_logits_larger_clap teacher_logits_fused \
        --logits_dirs_beats teacher_logits_beats \
        --n_cls 4

Output:
    Stacking ensemble ICBHI Score
"""
import os
import sys
import argparse
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def compute_icbhi_score(preds, labels, n_cls=4):
    hits = [0.0] * n_cls
    counts = [0.0] * n_cls
    for pred, label in zip(preds, labels):
        counts[label] += 1
        if pred == label:
            hits[label] += 1
    sp = hits[0] / (counts[0] + 1e-10) * 100
    se = sum(hits[1:]) / (sum(counts[1:]) + 1e-10) * 100
    return sp, se, (sp + se) / 2


def load_logits(logits_dir, split='test'):
    path = os.path.join(logits_dir, f'teacher_logits.{split}.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No logits found at {path}")
    data = torch.load(path, weights_only=False)
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, torch.Tensor):
        data = data.numpy()
    if data.ndim == 3:
        data = data.mean(axis=1)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logits_dirs', type=str, nargs='+', default=[])
    parser.add_argument('--logits_dirs_beats', type=str, nargs='+', default=[])
    parser.add_argument('--n_cls', type=int, default=4)
    parser.add_argument('--cv_folds', type=int, default=5)
    args = parser.parse_args()

    # Load labels
    train_data = torch.load('./data/training.pt', weights_only=False)
    train_labels = np.array([sample[1] for sample in train_data])
    test_data = torch.load('./data/test.pt', weights_only=False)
    test_labels = np.array([sample[1] for sample in test_data])

    # Load all logits
    model_names = []
    train_logits_list = []
    test_logits_list = []

    for d in args.logits_dirs + args.logits_dirs_beats:
        try:
            train_l = load_logits(d, 'training')
            test_l = load_logits(d, 'test')
            name = os.path.basename(d.rstrip('/'))
            model_names.append(name)
            train_logits_list.append(train_l)
            test_logits_list.append(test_l)
            print(f"Loaded: {name} (train={train_l.shape}, test={test_l.shape})")
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if not train_logits_list:
        print("Error: No logits loaded!")
        return

    # Concatenate all logits as features
    X_train = np.concatenate(train_logits_list, axis=1)  # [N_train, n_models * n_cls]
    X_test = np.concatenate(test_logits_list, axis=1)    # [N_test, n_models * n_cls]
    y_train = train_labels
    y_test = test_labels

    print(f"\nFeature shape: train={X_train.shape}, test={X_test.shape}")
    print(f"Models: {model_names}")

    # Cross-validated stacking
    print(f"\n--- {args.cv_folds}-Fold Cross-Validated Stacking ---")
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y_train))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        meta = LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial', random_state=42)
        meta.fit(X_tr, y_tr)
        oof_preds[val_idx] = meta.predict(X_val)

        fold_score = compute_icbhi_score(meta.predict(X_val), y_val, args.n_cls)
        print(f"  Fold {fold+1}: S_p={fold_score[0]:.2f}, S_e={fold_score[1]:.2f}, Score={fold_score[2]:.2f}")

    oof_score = compute_icbhi_score(oof_preds, y_train, args.n_cls)
    print(f"OOF: S_p={oof_score[0]:.2f}, S_e={oof_score[1]:.2f}, Score={oof_score[2]:.2f}")

    # Final model on full training set
    print("\n--- Final Stacking on Full Training Set ---")
    meta_final = LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial', random_state=42)
    meta_final.fit(X_train, y_train)

    test_preds = meta_final.predict(X_test)
    sp, se, sc = compute_icbhi_score(test_preds, y_test, args.n_cls)
    print(f"Stacking: S_p={sp:.2f}, S_e={se:.2f}, Score={sc:.2f}")

    # Compare with simple weighted average
    print("\n--- Comparison with Weighted Average ---")
    avg_probs = softmax(np.mean([softmax(l) for l in test_logits_list], axis=0), axis=1)
    avg_preds = np.argmax(avg_probs, axis=1)
    sp_avg, se_avg, sc_avg = compute_icbhi_score(avg_preds, test_labels, args.n_cls)
    print(f"Simple Average: S_p={sp_avg:.2f}, S_e={se_avg:.2f}, Score={sc_avg:.2f}")
    print(f"Stacking Gain: {sc - sc_avg:+.2f} points")

    # Save meta-learner
    import pickle
    os.makedirs('save', exist_ok=True)
    with open('save/stacking_meta_learner.pkl', 'wb') as f:
        pickle.dump(meta_final, f)
    print(f"\nMeta-learner saved to save/stacking_meta_learner.pkl")


if __name__ == '__main__':
    main()
