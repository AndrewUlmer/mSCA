"""
Compute BCV pseudo-R² vs n_components — nonlinear-nonlinear (MLP decoder).

Run from the project root:
    python experiments/reaching/R2_vs_ncomponents_nonlinear_decoder.py

Results saved as `bcv_scores.pt` in each n_components_K/ directory and
`bcv_summary.pt` at the condition level.
"""

import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from msca import mSCA
from msca.evaluations import evaluate_trial_average

DATA_PATH  = "./experiments/reaching/data/x_target_aligned.pt"
COND_DIR   = "./experiments/reaching/results/sweep-n_component/nonlinear-nonlinear"
OUT_DIR    = "./experiments/reaching/results/sweep-n_component/r2/nonlinear-nonlinear"
LOSS_FUNC  = "poisson"

N_COMPONENTS = [6, 10, 16, 20, 30, 40]

MODEL_KWARGS = dict(
    loss_func="Poisson", n_epochs=5000, post_hoc_epoch=-1,
    linear=False, lam_region=0.0,
    decoder_type="nonlinear", decoder_hidden_size=40, decoder_activation="tanh",
    cd_rate=0.5, cd_mode="both", filter_len=41,
    init="unique", decoder_init_mode="pca",
    sparsity_warmup_epochs=1000, balance_interval=1000,
)

print("Loading data...")
data = torch.load(DATA_PATH, weights_only=False)
X = {
    "M1":  [x.astype("float32") for x in data["M1"]],
    "PMd": [x.astype("float32") for x in data["PMd"]],
}
print(f"  M1:  {len(X['M1'])} trials, {X['M1'][0].shape[1]} neurons")
print(f"  PMd: {len(X['PMd'])} trials, {X['PMd'][0].shape[1]} neurons")

train_idxs = torch.load(os.path.join(COND_DIR, "n_train_splits.pt"), weights_only=False)
test_idxs  = torch.load(os.path.join(COND_DIR, "n_test_splits.pt"),  weights_only=False)
n_folds    = len(train_idxs[list(train_idxs.keys())[0]])

os.makedirs(OUT_DIR, exist_ok=True)
summary = {}

for K in N_COMPONENTS:
    model_dir   = os.path.join(COND_DIR, f"n_components_{K}")
    out_k_dir   = os.path.join(OUT_DIR, f"n_components_{K}")
    os.makedirs(out_k_dir, exist_ok=True)
    scores_path = os.path.join(out_k_dir, "bcv_scores.pt")

    if os.path.exists(scores_path):
        print(f"n_components={K}: loading cached scores")
        fold_scores = torch.load(scores_path, weights_only=False)
        summary[K] = fold_scores
        print(f"  scores={[f'{s:.4f}' for s in fold_scores]}  mean={np.mean(fold_scores):.4f}")
        continue

    print(f"\nn_components={K}")
    fold_scores = []

    for i in range(n_folds):
        X_train = {k: [trial[:, train_idxs[k][i]] for trial in v] for k, v in X.items()}
        X_val   = {k: [trial[:, test_idxs[k][i]]  for trial in v] for k, v in X.items()}

        model = mSCA(n_components=K, **MODEL_KWARGS)
        model.load(os.path.join(model_dir, f"msca_split_{i}.pt"), X_train)

        perfs = evaluate_trial_average(
            model, X_train, X_val,
            loss_func=LOSS_FUNC,
            decoder_type="nonlinear",
            n_splits=5,
        )
        score = float(np.mean(perfs))
        fold_scores.append(score)
        print(f"  fold {i}: {score:.4f}")

    torch.save(fold_scores, scores_path)
    summary[K] = fold_scores
    print(f"  → mean={np.mean(fold_scores):.4f}  saved to {scores_path}")

summary_path = os.path.join(OUT_DIR, "bcv_summary.pt")
torch.save(summary, summary_path)
print(f"\nSummary saved → {summary_path}")
for K, scores in summary.items():
    print(f"  n_components={K:2d}: mean={np.mean(scores):.4f} ± {np.std(scores):.4f}")

print("\nDone.")
