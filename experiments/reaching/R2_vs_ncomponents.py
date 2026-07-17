"""
Compute BCV pseudo-R² vs n_components for the sweep-n_component results.

Run from the project root:
    python experiments/reaching/R2_vs_ncomponents.py

Results are saved as `bcv_scores.pt` inside each
    results/sweep-n_component/<decoder-type>/n_components_<K>/
directory, and a summary `bcv_summary.pt` at the decoder-type level.
"""

import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from msca import mSCA
from msca.evaluations import evaluate_trial_average

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH   = "./experiments/reaching/data/x_target_aligned.pt"
SWEEP_ROOT  = "./experiments/reaching/results/sweep-n_component"

N_COMPONENTS = [6, 10, 16, 20, 30, 40]
LOSS_FUNC    = "poisson"

CONDITIONS = {
    "nonlinear-nonlinear": {
        "decoder_type": "nonlinear",
        "model_kwargs": dict(
            loss_func="Poisson", n_epochs=5000, post_hoc_epoch=-1,
            linear=False, lam_region=0.0,
            decoder_type="nonlinear", decoder_hidden_size=40, decoder_activation="tanh",
            cd_rate=0.5, cd_mode="both", filter_len=41,
            init="unique", decoder_init_mode="pca",
            sparsity_warmup_epochs=1000, balance_interval=1000,
        ),
    },
    "nonlinear-linear": {
        "decoder_type": "linear",
        "model_kwargs": dict(
            loss_func="Poisson", n_epochs=5000, post_hoc_epoch=-1,
            linear=False, lam_region=0.0,
            decoder_type="linear", decoder_hidden_size=40, decoder_activation="tanh",
            cd_rate=0.5, cd_mode="both", filter_len=41,
            init="unique", decoder_init_mode="pca",
            sparsity_warmup_epochs=1000, balance_interval=1000,
        ),
    },
}

# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading data...")
data = torch.load(DATA_PATH, weights_only=False)
X = {
    "M1":  [x.astype("float32") for x in data["M1"]],
    "PMd": [x.astype("float32") for x in data["PMd"]],
}
print(f"  M1:  {len(X['M1'])} trials, {X['M1'][0].shape[1]} neurons")
print(f"  PMd: {len(X['PMd'])} trials, {X['PMd'][0].shape[1]} neurons")

# ── Evaluate ──────────────────────────────────────────────────────────────────

for cond_name, cond in CONDITIONS.items():
    cond_dir      = os.path.join(SWEEP_ROOT, cond_name)
    decoder_type  = cond["decoder_type"]
    model_kwargs  = cond["model_kwargs"]

    train_idxs = torch.load(os.path.join(cond_dir, "n_train_splits.pt"), weights_only=False)
    test_idxs  = torch.load(os.path.join(cond_dir, "n_test_splits.pt"),  weights_only=False)
    n_folds    = len(train_idxs[list(train_idxs.keys())[0]])

    summary = {}  # n_components → list of per-fold mean scores

    for K in N_COMPONENTS:
        model_dir   = os.path.join(cond_dir, f"n_components_{K}")
        scores_path = os.path.join(model_dir, "bcv_scores.pt")

        if os.path.exists(scores_path):
            print(f"[{cond_name}] n_components={K}: loading cached scores")
            fold_scores = torch.load(scores_path, weights_only=False)
            summary[K] = fold_scores
            print(f"  scores={[f'{s:.4f}' for s in fold_scores]}  mean={np.mean(fold_scores):.4f}")
            continue

        print(f"\n[{cond_name}] n_components={K}")
        fold_scores = []

        for i in range(n_folds):
            X_train = {k: [trial[:, train_idxs[k][i]] for trial in v] for k, v in X.items()}
            X_val   = {k: [trial[:, test_idxs[k][i]]  for trial in v] for k, v in X.items()}

            model = mSCA(n_components=K, **model_kwargs)
            model.load(os.path.join(model_dir, f"msca_split_{i}.pt"), X_train)

            perfs = evaluate_trial_average(
                model, X_train, X_val,
                loss_func=LOSS_FUNC,
                decoder_type=decoder_type,
                n_splits=5,
            )
            score = float(np.mean(perfs))
            fold_scores.append(score)
            print(f"  fold {i}: {score:.4f}")

        torch.save(fold_scores, scores_path)
        summary[K] = fold_scores
        print(f"  → mean={np.mean(fold_scores):.4f}  saved to {scores_path}")

    summary_path = os.path.join(cond_dir, "bcv_summary.pt")
    torch.save(summary, summary_path)
    print(f"\n[{cond_name}] Summary saved → {summary_path}")
    for K, scores in summary.items():
        print(f"  n_components={K:2d}: mean={np.mean(scores):.4f} ± {np.std(scores):.4f}")

print("\nDone.")
