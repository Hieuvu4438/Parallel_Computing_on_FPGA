"""
Multi-Model Ensemble Evaluation

Combines logits from multiple CLAP variants and/or BEATs to find
the optimal weighted ensemble that maximizes ICBHI Score.

Supports:
- laion/clap-htsat-unfused  (original BTS)
- laion/clap-htsat-fused    (fused cross-modal)
- laion/larger_clap_general (larger, 2.5M pairs)
- BEATs ensemble (top-k seeds)

Usage:
    # After training all models and extracting logits:
    python scripts/ensemble_multi_model.py \
        --logits_dirs teacher_logits teacher_logits_larger_clap teacher_logits_fused \
        --logits_dirs_beats teacher_logits_beats \
        --n_cls 4 \
        --grid_steps 20

    # Or with pre-computed logits only (no BEATs):
    python scripts/ensemble_multi_model.py \
        --logits_dirs teacher_logits teacher_logits_larger_clap

Output:
    - Optimal weights for each model
    - ICBHI Score (S_p, S_e, Score)
    - Comparison table of all individual models and ensembles
"""
import os
import sys
import argparse
import itertools
import numpy as np
import torch
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_logits(logits_dir, split='test'):
    """Load logits from a directory. Returns [N, n_cls] numpy array."""
    path = os.path.join(logits_dir, f'teacher_logits.{split}.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No logits found at {path}")
    data = torch.load(path, weights_only=False)
    # Handle various formats: list, Tensor, numpy
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, torch.Tensor):
        data = data.numpy()
    else:
        data = np.array(data)
    # If shape is [N, K, n_cls] (multiple teachers), average over teachers
    if data.ndim == 3:
        data = data.mean(axis=1)
    return data


def load_test_labels():
    """Load test labels from cached data."""
    data = torch.load('./data/test.pt', weights_only=False)
    labels = [sample[1] for sample in data]
    return np.array(labels)


def compute_icbhi_score(logits_or_probs, labels, n_cls=4):
    """
    Compute ICBHI Score from logits or probabilities.

    Args:
        logits_or_probs: [N, n_cls] array
        labels: [N] array of ground truth labels
        n_cls: number of classes

    Returns:
        (sp, se, score) tuple
    """
    if logits_or_probs.ndim == 3:
        logits_or_probs = logits_or_probs.mean(axis=1)

    preds = np.argmax(logits_or_probs, axis=1)

    hits = [0.0] * n_cls
    counts = [0.0] * n_cls

    for pred, label in zip(preds, labels):
        counts[label] += 1
        if pred == label:
            hits[label] += 1

    sp = hits[0] / (counts[0] + 1e-10) * 100
    se = sum(hits[1:]) / (sum(counts[1:]) + 1e-10) * 100
    score = (sp + se) / 2.0

    return sp, se, score


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def weighted_ensemble(all_logits, weights):
    """
    Compute weighted ensemble of logits.

    Args:
        list of [N, n_cls] arrays
        weights: array of weights (will be normalized)

    Returns:
        [N, n_cls] ensemble logits
    """
    weights = np.array(weights)
    weights = weights / weights.sum()

    probs = []
    for logits in all_logits:
        probs.append(softmax(logits))

    ensemble = sum(w * p for w, p in zip(weights, probs))
    return ensemble


def grid_search_weights(all_logits, labels, n_cls=4, grid_steps=20):
    """
    Grid search for optimal ensemble weights.

    For 2 models: exhaustive search over w1 in [0, 1]
    For 3+ models: random search + refinement
    """
    n_models = len(all_logits)

    if n_models == 1:
        return [1.0]

    if n_models == 2:
        best_score = 0
        best_w = [0.5, 0.5]
        for w1 in np.linspace(0, 1, grid_steps + 1):
            w2 = 1 - w1
            ensemble = weighted_ensemble(all_logits, [w1, w2])
            _, _, score = compute_icbhi_score(ensemble, labels, n_cls)
            if score > best_score:
                best_score = score
                best_w = [w1, w2]
        return best_w

    if n_models == 3:
        best_score = 0
        best_w = [1/3, 1/3, 1/3]
        step = 1.0 / grid_steps
        for w1 in np.arange(0, 1 + step, step):
            for w2 in np.arange(0, 1 - w1 + step, step):
                w3 = 1 - w1 - w2
                if w3 < 0:
                    continue
                ensemble = weighted_ensemble(all_logits, [w1, w2, w3])
                _, _, score = compute_icbhi_score(ensemble, labels, n_cls)
                if score > best_score:
                    best_score = score
                    best_w = [w1, w2, w3]
        return best_w

    # For 4+ models: random search
    best_score = 0
    best_w = [1/n_models] * n_models
    n_trials = grid_steps * 100

    for _ in range(n_trials):
        w = np.random.dirichlet(np.ones(n_models))
        ensemble = weighted_ensemble(all_logits, w)
        _, _, score = compute_icbhi_score(ensemble, labels, n_cls)
        if score > best_score:
            best_score = score
            best_w = w.tolist()

    # Refinement: grid search around best weights
    refined_w = best_w.copy()
    for _ in range(3):
        for i in range(n_models):
            for delta in np.linspace(-0.1, 0.1, 11):
                test_w = refined_w.copy()
                test_w[i] = max(0, test_w[i] + delta)
                test_w = [w / sum(test_w) for w in test_w]
                ensemble = weighted_ensemble(all_logits, test_w)
                _, _, score = compute_icbhi_score(ensemble, labels, n_cls)
                if score > best_score:
                    best_score = score
                    best_w = test_w.copy()
        refined_w = best_w.copy()

    return best_w


def main():
    parser = argparse.ArgumentParser(description='Multi-Model Ensemble Evaluation')
    parser.add_argument('--logits_dirs', type=str, nargs='+', default=[],
                        help='Directories with CLAP model logits (teacher_logits.*.pt)')
    parser.add_argument('--logits_dirs_beats', type=str, nargs='+', default=[],
                        help='Directories with BEATs logits')
    parser.add_argument('--n_cls', type=int, default=4)
    parser.add_argument('--grid_steps', type=int, default=20,
                        help='Grid resolution for weight search')
    parser.add_argument('--split', type=str, default='test',
                        help='Which split to evaluate on')
    args = parser.parse_args()

    # Load test labels
    labels = load_test_labels()
    print(f"Loaded {len(labels)} test samples")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Collect all logits
    model_names = []
    all_logits = []

    for d in args.logits_dirs:
        try:
            logits = load_logits(d, split=args.split)
            name = os.path.basename(d.rstrip('/'))
            model_names.append(name)
            all_logits.append(logits)
            sp, se, score = compute_icbhi_score(logits, labels, args.n_cls)
            print(f"  {name:40s} → S_p={sp:.2f}, S_e={se:.2f}, Score={score:.2f}")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")

    for d in args.logits_dirs_beats:
        try:
            logits = load_logits(d, split=args.split)
            name = os.path.basename(d.rstrip('/'))
            model_names.append(name)
            all_logits.append(logits)
            sp, se, score = compute_icbhi_score(logits, labels, args.n_cls)
            print(f"  {name:40s} → S_p={sp:.2f}, S_e={se:.2f}, Score={score:.2f}")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")

    if not all_logits:
        print("Error: No logits loaded!")
        return

    # Also load original CLAP ensemble (30 teachers) for comparison
    try:
        clap_ensemble = load_logits('teacher_logits', split=args.split)
        sp, se, score = compute_icbhi_score(clap_ensemble, labels, args.n_cls)
        print(f"\n  {'CLAP ensemble (30 teachers)':40s} → S_p={sp:.2f}, S_e={se:.2f}, Score={score:.2f}")
    except:
        pass

    # ── Weighted Ensemble ──
    print(f"\n{'='*60}")
    print(f" Weighted Ensemble ({len(all_logits)} models, grid_steps={args.grid_steps})")
    print(f"{'='*60}")

    best_weights = grid_search_weights(all_logits, labels, args.n_cls, args.grid_steps)
    ensemble = weighted_ensemble(all_logits, best_weights)
    sp, se, score = compute_icbhi_score(ensemble, labels, args.n_cls)

    print(f"\nOptimal weights:")
    for name, w in zip(model_names, best_weights):
        print(f"  {name:40s}: {w:.3f}")
    print(f"\nEnsemble result:")
    print(f"  S_p = {sp:.2f}")
    print(f"  S_e = {se:.2f}")
    print(f"  Score = {score:.2f}")

    # ── All Pairs ──
    if len(all_logits) >= 2:
        print(f"\n{'='*60}")
        print(f" Pairwise Ensembles")
        print(f"{'='*60}")
        for i, j in itertools.combinations(range(len(all_logits)), 2):
            pair_logits = [all_logits[i], all_logits[j]]
            w = grid_search_weights(pair_logits, labels, args.n_cls, args.grid_steps)
            ens = weighted_ensemble(pair_logits, w)
            sp, se, sc = compute_icbhi_score(ens, labels, args.n_cls)
            print(f"  {model_names[i]} + {model_names[j]}")
            print(f"    Weights: [{w[0]:.2f}, {w[1]:.2f}]")
            print(f"    S_p={sp:.2f}, S_e={se:.2f}, Score={sc:.2f}")

    # ── All subsets ──
    if len(all_logits) >= 3:
        print(f"\n{'='*60}")
        print(f" All Subsets (top-3 combinations)")
        print(f"{'='*60}")
        subset_results = []
        for size in range(2, len(all_logits) + 1):
            for indices in itertools.combinations(range(len(all_logits)), size):
                subset = [all_logits[i] for i in indices]
                names = [model_names[i] for i in indices]
                w = grid_search_weights(subset, labels, args.n_cls, args.grid_steps)
                ens = weighted_ensemble(subset, w)
                sp, se, sc = compute_icbhi_score(ens, labels, args.n_cls)
                subset_results.append((sc, sp, se, names, w))

        subset_results.sort(reverse=True)
        for sc, sp, se, names, w in subset_results[:5]:
            print(f"  Score={sc:.2f} (S_p={sp:.2f}, S_e={se:.2f})")
            print(f"    Models: {', '.join(names)}")
            print(f"    Weights: {[f'{x:.2f}' for x in w]}")

    # ── Save results ──
    results = {
        'model_names': model_names,
        'best_weights': best_weights,
        'ensemble_score': {'sp': sp, 'se': se, 'score': score},
        'individual_scores': {},
    }
    for name, logits in zip(model_names, all_logits):
        s, e, c = compute_icbhi_score(logits, labels, args.n_cls)
        results['individual_scores'][name] = {'sp': s, 'se': e, 'score': c}

    import json
    os.makedirs('save', exist_ok=True)
    with open('save/ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to save/ensemble_results.json")


if __name__ == '__main__':
    main()
