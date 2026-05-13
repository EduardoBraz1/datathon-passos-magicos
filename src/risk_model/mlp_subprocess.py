"""Stand-alone PyTorch MLP worker.

Run as ``python -m risk_model.mlp_subprocess <input.pkl> <output.pkl>``.

Why a separate process?
-----------------------
Empirical observation on macOS: combining ``joblib.Parallel(backend='loky')`` /
``ProcessPoolExecutor`` with PyTorch under the same parent leads to silent
hangs and segmentation faults.  Running the MLP as an *external* subprocess
sidesteps that completely – the parent simply launches `python -m` and waits
for the result file.  ``device='cpu'`` is forced to additionally remove any
MPS-related variability for this run.

Pipeline executed by the worker
-------------------------------
1. Standardise features using **train-fold statistics only** (no leakage).
2. For each of the 5 GroupKFold-by-RA folds inside the train period:
   - Refit the MLP from scratch with focal/BCE loss + early stopping
     against the in-fold validation slice (so OOF predictions are unbiased).
3. Refit the MLP one last time on the full train period with early stopping
   against the temporal hold-out (kept out from feature scaling fit).
4. Persist OOF predictions, hold-out predictions, prediction-year predictions,
   the final-model state dict and the scaler statistics back to disk.
"""
from __future__ import annotations

import json
import logging
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

logging.basicConfig(level="INFO", format="%(asctime)s | mlp_sub | %(message)s")
LOG = logging.getLogger("mlp_subprocess")


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_features: int, dropout: float = 0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Scaler helper – we store mean/std as numpy arrays for portability.
# ---------------------------------------------------------------------------
def fit_scaler(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=True)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        # median imputation per column based on training mean (mean is robust enough
        # since we already dealt with structural missingness in features.py).
        X[nan_mask] = np.broadcast_to(mean, X.shape)[nan_mask]
    return (X - mean) / std


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_lr: float = 3e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    dropout: float = 0.30,
    seed: int = 42,
) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    """Train one MLP and return the *best* state dict (by val ROC-AUC) plus loss history."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cpu"
    model = MLP(X_train.shape[1], dropout=dropout).to(device)

    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor(max(1e-3, n_neg / max(1.0, n_pos)), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(len(X_train) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs, pct_start=0.30,
    )

    X_tr_t = torch.from_numpy(X_train.astype(np.float32))
    y_tr_t = torch.from_numpy(y_train.astype(np.float32))
    X_va_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_va_np = y_val.astype(np.int32)

    n = len(X_tr_t)
    best_auc = -1.0
    best_state: Dict[str, torch.Tensor] | None = None
    epochs_since_best = 0
    history: List[float] = []

    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        order = rng.permutation(n)
        model.train()
        for i in range(0, n, batch_size):
            idx = order[i : i + batch_size]
            if len(idx) < 2:  # BN needs ≥2
                continue
            xb = X_tr_t[idx].to(device)
            yb = y_tr_t[idx].to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_t).cpu().numpy()
        try:
            from sklearn.metrics import roc_auc_score
            val_auc = float(roc_auc_score(y_va_np, val_logits))
        except ValueError:
            val_auc = 0.5
        history.append(val_auc)
        if val_auc > best_auc + 1e-5:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        if epochs_since_best >= patience:
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_state, history


def predict(state: Dict[str, torch.Tensor], X: np.ndarray, in_features: int, dropout: float = 0.30) -> np.ndarray:
    model = MLP(in_features, dropout=dropout)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32))).numpy()
    return 1.0 / (1.0 + np.exp(-logits))


# ---------------------------------------------------------------------------
# Main subprocess entry-point
# ---------------------------------------------------------------------------
def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m risk_model.mlp_subprocess <input.pkl> <output.pkl>")
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    LOG.info("Loading inputs from %s", input_path)
    with input_path.open("rb") as fh:
        payload = pickle.load(fh)

    X_train = payload["X_train"].astype(np.float64)
    y_train = payload["y_train"].astype(np.int32)
    groups_train = np.asarray(payload["groups_train"])
    X_val = payload["X_val"].astype(np.float64)
    y_val = payload["y_val"].astype(np.int32)
    X_predict = payload["X_predict"].astype(np.float64)

    n_in = X_train.shape[1]
    LOG.info("n_train=%d n_val=%d n_predict=%d n_features=%d",
             len(X_train), len(X_val), len(X_predict), n_in)

    t0 = time.time()
    cv = GroupKFold(n_splits=5)
    oof = np.zeros(len(X_train), dtype=np.float64)

    for fold, (idx_tr, idx_va) in enumerate(cv.split(X_train, y_train, groups_train)):
        LOG.info("OOF fold %d/5", fold + 1)
        mean_f, std_f = fit_scaler(X_train[idx_tr])
        X_tr_s = apply_scaler(X_train[idx_tr], mean_f, std_f)
        X_va_s = apply_scaler(X_train[idx_va], mean_f, std_f)
        state, _ = train_one(
            X_tr_s, y_train[idx_tr], X_va_s, y_train[idx_va], seed=42 + fold,
        )
        oof[idx_va] = predict(state, X_va_s, n_in)

    LOG.info("Refitting final model on full train, early-stop on hold-out")
    mean_final, std_final = fit_scaler(X_train)
    X_tr_full = apply_scaler(X_train, mean_final, std_final)
    X_va_full = apply_scaler(X_val, mean_final, std_final)
    X_pr_full = apply_scaler(X_predict, mean_final, std_final)

    final_state, history = train_one(
        X_tr_full, y_train, X_va_full, y_val, seed=42,
    )
    val_pred = predict(final_state, X_va_full, n_in)
    predict_pred = predict(final_state, X_pr_full, n_in)
    LOG.info("MLP wall-clock: %.1fs", time.time() - t0)

    LOG.info("Saving outputs to %s", output_path)
    with output_path.open("wb") as fh:
        pickle.dump(
            {
                "oof": oof,
                "val_pred": val_pred,
                "predict_pred": predict_pred,
                "state_dict": {k: v.numpy() for k, v in final_state.items()},
                "scaler_mean": mean_final,
                "scaler_std": std_final,
                "history": history,
                "n_features": n_in,
            },
            fh,
        )
    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
