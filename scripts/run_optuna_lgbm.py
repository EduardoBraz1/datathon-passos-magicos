"""Bayesian search (Optuna, TPE) for LightGBM on the risk-composite task.

Search runs ONLY on the 2022→2023 training fold using GroupKFold-5 by RA;
the 2023→2024 hold-out is touched **once** at the end with the winning
hyper-parameters to produce honest ANTES vs DEPOIS numbers.

Outputs:
- ``experiments/<timestamp>/optuna_lgbm/study.csv``  — every trial logged.
- ``experiments/<timestamp>/optuna_lgbm/best.json``  — best params + scores.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_pipeline import assemble, LGBM_BASE  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
LOG = logging.getLogger("optuna_lgbm")
optuna.logging.set_verbosity(optuna.logging.WARNING)


SEEDS = (42, 7, 2024)


def cv_auc(params: dict, X, y, groups, n_estimators_max: int = 1500) -> float:
    cv = GroupKFold(n_splits=5)
    aucs = []
    for idx_tr, idx_va in cv.split(X, y, groups):
        full = dict(LGBM_BASE)
        full.update(params)
        full["n_estimators"] = n_estimators_max
        m = lgb.LGBMClassifier(**full)
        m.fit(
            X.iloc[idx_tr], y.iloc[idx_tr],
            eval_set=[(X.iloc[idx_va], y.iloc[idx_va])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        proba = m.predict_proba(X.iloc[idx_va])[:, 1]
        aucs.append(roc_auc_score(y.iloc[idx_va], proba))
    return float(np.mean(aucs))


def objective(trial: optuna.Trial, X, y, groups) -> float:
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [0, 1, 3, 5]),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 5.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 5.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
    }
    return cv_auc(params, X, y, groups)


def best_iter_search(params: dict, X, y, groups) -> int:
    """Estimate the average best iteration across folds for the chosen params."""
    cv = GroupKFold(n_splits=5)
    iters = []
    for idx_tr, idx_va in cv.split(X, y, groups):
        full = dict(LGBM_BASE)
        full.update(params)
        full["n_estimators"] = 2000
        m = lgb.LGBMClassifier(**full)
        m.fit(
            X.iloc[idx_tr], y.iloc[idx_tr],
            eval_set=[(X.iloc[idx_va], y.iloc[idx_va])],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
        iters.append(int(m.best_iteration_ or 200))
    return int(np.mean(iters))


def main(n_trials: int = 50) -> int:
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = ROOT / "experiments" / ts / "optuna_lgbm"
    out.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading datasets…")
    ds = assemble()
    LOG.info("Train n=%d (pos=%.2f%%) | features=%d", len(ds.X_train),
             100 * ds.y_train.mean(), len(ds.feature_cols))

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name="lgbm_risk_composite")

    t0 = time.time()
    study.optimize(
        lambda t: objective(t, ds.X_train, ds.y_train, ds.groups_train),
        n_trials=n_trials,
        show_progress_bar=False,
        catch=(Exception,),
    )
    dt = time.time() - t0

    LOG.info("Optuna best CV-AUC=%.4f (%d trials, %.1fs)",
             study.best_value, n_trials, dt)
    LOG.info("Best params: %s", study.best_params)

    # Re-fit with seed-bagging and average best_iter, evaluate ONCE on hold-out.
    best_params = study.best_params
    n_est = best_iter_search(best_params, ds.X_train, ds.y_train, ds.groups_train)
    LOG.info("Average best_iteration across folds: %d", n_est)

    val_probas = []
    predict_probas = []
    for seed in SEEDS:
        full = dict(LGBM_BASE)
        full.update(best_params)
        full["random_state"] = seed
        full["n_estimators"] = max(50, min(int(n_est * 1.1), 2000))
        m = lgb.LGBMClassifier(**full)
        # Important: do NOT early-stop on the hold-out — that would peek.
        m.fit(ds.X_train, ds.y_train)
        val_probas.append(m.predict_proba(ds.X_val)[:, 1])
        predict_probas.append(m.predict_proba(ds.X_predict)[:, 1])
    val_proba = np.mean(val_probas, axis=0)
    predict_proba = np.mean(predict_probas, axis=0)

    import pandas as pd
    preds_path = ROOT / "data" / "processed" / "predicoes_risco_2025.csv"
    if preds_path.exists():
        existing = pd.read_csv(preds_path)
        ra_to_exp = dict(zip(ds.predict_ra.astype(str), predict_proba))
        existing["prob_experimental"] = existing["RA"].astype(str).map(ra_to_exp).round(4)
        existing.to_csv(preds_path, index=False)
        LOG.info("Added prob_experimental column to %s", preds_path)

    roc = roc_auc_score(ds.y_val, val_proba)
    pr = average_precision_score(ds.y_val, val_proba)
    bs = brier_score_loss(ds.y_val, val_proba)
    # Use train OOF threshold = 0.29 (matches main pipeline).
    thr = 0.29
    f1 = f1_score(ds.y_val, (val_proba >= thr).astype(int))
    top10 = int(len(ds.y_val) * 0.10)
    order = np.argsort(-val_proba)[:top10]
    recall_top = float(ds.y_val.iloc[order].sum() / max(1, ds.y_val.sum()))

    holdout = {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "brier": float(bs),
        "f1_at_0.29": float(f1),
        "recall_top10pct": recall_top,
    }
    LOG.info("Optuna LGBM hold-out: %s", holdout)

    summary = {
        "timestamp": ts,
        "n_trials": n_trials,
        "search_seconds": dt,
        "best_cv_auc": float(study.best_value),
        "best_params": best_params,
        "avg_best_iteration": n_est,
        "holdout": holdout,
    }
    (out / "best.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    trials_df = study.trials_dataframe()
    trials_df.to_csv(out / "study.csv", index=False)
    LOG.info("Artifacts: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(n_trials=50))
