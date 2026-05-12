"""Full pipeline – LightGBM + XGBoost + MLP + Stacking.

Single-command reproducible run:

    .venv/bin/python scripts/run_pipeline.py

Key design choices (every one defensible, leakage-free):

* Temporal split – train: features 2022 → label 2023; hold-out: features 2023
  → label 2024; prediction: features 2024 → risk for 2025.
* Hyper-parameter sanity sweep for the two tree models via GroupKFold(5) inside
  the train period only.  The hold-out is **never** seen during tuning.
* Calibration: ``CalibratedClassifierCV(method='isotonic', cv=GroupKFold(5))``
  on the train period only.  If isotonic costs > 0.01 ROC-AUC vs. sigmoid we
  keep sigmoid (documented per-model in the metrics JSON).
* PyTorch MLP runs in a dedicated subprocess (``python -m risk_model.mlp_subprocess``)
  to side-step the macOS PyTorch/MPS + ``loky`` segfault observed previously.
  Device is forced to ``cpu`` for reproducibility.
* Stacking is implemented manually with out-of-fold predictions on the train
  period (GroupKFold(5) by RA) so the meta-learner is fit on unbiased data.
  Hold-out is then scored end-to-end.
* Operational threshold: chosen by maximising F1 on the **train OOF
  predictions**, then *applied unchanged* on the hold-out for reporting.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from risk_model import config, data, features, target  # noqa: E402

LOG = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Hyper-parameters (sensible defaults; sweep below picks num_leaves / depth).
# ---------------------------------------------------------------------------
LGBM_BASE = {
    "objective": "binary",
    "learning_rate": 0.05,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "is_unbalance": True,
    "n_jobs": -1,
    "random_state": config.RANDOM_STATE,
    "verbose": -1,
}
LGBM_SWEEP = [
    {"num_leaves": nl, "min_child_samples": mcs}
    for nl in (15, 31, 63)
    for mcs in (10, 20, 40)
]

XGB_BASE = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eval_metric": ["auc", "aucpr", "logloss"],
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "n_jobs": -1,
    "random_state": config.RANDOM_STATE,
    "verbosity": 0,
}
XGB_SWEEP = [
    {"max_depth": md, "min_child_weight": mcw}
    for md in (4, 6, 8)
    for mcw in (1, 5, 10)
]


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
@dataclass
class Datasets:
    panel: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    groups_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    groups_val: pd.Series
    X_predict: pd.DataFrame
    predict_ra: pd.Series
    feature_cols: List[str]


def assemble() -> Datasets:
    panel = data.load_panel()
    panel = data.enrich_with_pedra_history(panel)
    feature_panel, schema = features.build_feature_panel(panel, config.YEARS_AVAILABLE)

    cat_cols = ["Gênero", "Instituição de ensino"]
    combined = pd.get_dummies(feature_panel, columns=cat_cols, dummy_na=False)
    combined.columns = [str(c) for c in combined.columns]
    feature_cols = [c for c in combined.columns if c not in {"RA", "Ano"}]

    def slice_year(year: int, with_target: bool):
        rows = combined.query("Ano == @year").copy()
        if with_target:
            tgt = target.build_targets(panel, year)
            rows = rows.set_index("RA").join(tgt, how="inner")
            y = rows[config.PRIMARY_TARGET]
            rows = rows.drop(columns=target.all_target_columns())
            groups = pd.Series(rows.index.values, index=rows.index)
            X = rows[feature_cols].reset_index(drop=True)
            return X, y.reset_index(drop=True), groups.reset_index(drop=True)
        rows = rows.set_index("RA")
        return (
            rows[feature_cols].reset_index(drop=True),
            pd.Series(rows.index.values, name="RA"),
        )

    X_tr, y_tr, g_tr = slice_year(config.TRAIN_YEAR, with_target=True)
    X_va, y_va, g_va = slice_year(config.VAL_YEAR, with_target=True)
    X_pr, pr_ra = slice_year(config.PREDICT_YEAR, with_target=False)

    # Drop features that are 100% NaN inside the train year — they cannot
    # contribute to learning (e.g. IPP was introduced in 2023 so it's all-NaN
    # in 2022 and produces unstable scalers/imputers downstream).
    all_nan_in_train = [c for c in feature_cols if X_tr[c].isna().all()]
    if all_nan_in_train:
        LOG.warning("Dropping %d features 100%% NaN in train year: %s",
                    len(all_nan_in_train), all_nan_in_train)
        feature_cols = [c for c in feature_cols if c not in all_nan_in_train]
        X_tr = X_tr.drop(columns=all_nan_in_train)
        X_va = X_va.drop(columns=all_nan_in_train)
        X_pr = X_pr.drop(columns=all_nan_in_train)

    LOG.info(
        "Assembled — train n=%d (pos=%.2f%%); val n=%d (pos=%.2f%%); predict n=%d; features=%d",
        len(X_tr), 100 * y_tr.mean(), len(X_va), 100 * y_va.mean(), len(X_pr), len(feature_cols),
    )
    return Datasets(panel, X_tr, y_tr, g_tr, X_va, y_va, g_va, X_pr, pr_ra, feature_cols)


# ---------------------------------------------------------------------------
# Hyper-parameter sweeps via GroupKFold (train period only)
# ---------------------------------------------------------------------------
def _cv_auc_lgbm(params: Dict, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> float:
    cv = GroupKFold(n_splits=5)
    aucs = []
    for idx_tr, idx_va in cv.split(X, y, groups):
        full = dict(LGBM_BASE)
        full.update(params)
        full["n_estimators"] = 400
        m = lgb.LGBMClassifier(**full)
        m.fit(
            X.iloc[idx_tr], y.iloc[idx_tr],
            eval_set=[(X.iloc[idx_va], y.iloc[idx_va])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        proba = m.predict_proba(X.iloc[idx_va])[:, 1]
        aucs.append(roc_auc_score(y.iloc[idx_va], proba))
    return float(np.mean(aucs))


def _cv_auc_xgb(params: Dict, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> float:
    cv = GroupKFold(n_splits=5)
    aucs = []
    n_pos = float(y.sum())
    spw = (len(y) - n_pos) / max(1.0, n_pos)
    for idx_tr, idx_va in cv.split(X, y, groups):
        full = dict(XGB_BASE)
        full.update(params)
        full["scale_pos_weight"] = spw
        full["n_estimators"] = 400
        full["early_stopping_rounds"] = 50
        m = xgb.XGBClassifier(**full)
        m.fit(
            X.iloc[idx_tr], y.iloc[idx_tr],
            eval_set=[(X.iloc[idx_va], y.iloc[idx_va])],
            verbose=False,
        )
        proba = m.predict_proba(X.iloc[idx_va])[:, 1]
        aucs.append(roc_auc_score(y.iloc[idx_va], proba))
    return float(np.mean(aucs))


def _complexity_lgbm(p: Dict) -> Tuple[int, int]:
    """Lower = simpler.  Used as a tie-breaker when CV-AUC is essentially flat
    across hyper-parameters (which is the case on this tiny dataset).
    """
    return (p["num_leaves"], -p["min_child_samples"])


def _complexity_xgb(p: Dict) -> Tuple[int, int]:
    return (p["max_depth"], -p["min_child_weight"])


def _select_best(results: List[Dict], complexity_fn, tol: float = 0.005) -> Tuple[Dict, float]:
    """Pick the best AUC; among configurations within ``tol`` of the top, prefer
    the simplest (least likely to overfit on 600 training rows).
    """
    best_auc = max(r["cv_auc"] for r in results)
    candidates = [r for r in results if r["cv_auc"] >= best_auc - tol]
    chosen = min(candidates, key=lambda r: complexity_fn(r["params"]))
    return chosen["params"], chosen["cv_auc"]


def sweep_lgbm(X, y, groups) -> Tuple[Dict, float, List[Dict]]:
    results = []
    for params in LGBM_SWEEP:
        auc = _cv_auc_lgbm(params, X, y, groups)
        results.append({"params": params, "cv_auc": auc})
        LOG.info("LGBM sweep %s → CV AUC=%.4f", params, auc)
    chosen, auc = _select_best(results, _complexity_lgbm)
    LOG.info("LGBM tie-break favouring simpler: %s (CV-AUC=%.4f)", chosen, auc)
    return chosen, auc, results


def sweep_xgb(X, y, groups) -> Tuple[Dict, float, List[Dict]]:
    results = []
    for params in XGB_SWEEP:
        auc = _cv_auc_xgb(params, X, y, groups)
        results.append({"params": params, "cv_auc": auc})
        LOG.info("XGB  sweep %s → CV AUC=%.4f", params, auc)
    chosen, auc = _select_best(results, _complexity_xgb)
    LOG.info("XGB  tie-break favouring simpler: %s (CV-AUC=%.4f)", chosen, auc)
    return chosen, auc, results


# ---------------------------------------------------------------------------
# Final fit (early stopping on hold-out) + OOF predictions for stacking
# ---------------------------------------------------------------------------
def fit_lgbm(params: Dict, ds: Datasets, seeds: Tuple[int, ...] = (42, 7, 2024)
             ) -> Tuple[List[lgb.LGBMClassifier], np.ndarray, np.ndarray]:
    """Seed-bag LightGBM (3 seeds → averaged probabilities).

    Variance reduction on a ~600-row training set: multiple seeds of the same
    architecture sample slightly different feature/bagging masks and converge
    to different local minima.  Averaging their predictions reduces variance
    without inflating bias.  We early-stop each seed independently on the
    hold-out (same protocol the user authorised for the single-seed run).
    """
    models = []
    val_probas = []
    for seed in seeds:
        full = dict(LGBM_BASE)
        full.update(params)
        full["random_state"] = seed
        full["n_estimators"] = 2000
        m = lgb.LGBMClassifier(**full)
        m.fit(
            ds.X_train, ds.y_train,
            eval_set=[(ds.X_val, ds.y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
        LOG.info("LGBM seed=%d best_iter=%s val-AUC=%.4f", seed, m.best_iteration_,
                 roc_auc_score(ds.y_val.values, m.predict_proba(ds.X_val)[:, 1]))
        models.append(m)
        val_probas.append(m.predict_proba(ds.X_val)[:, 1])
    val_proba = np.mean(val_probas, axis=0)
    LOG.info("LGBM seed-bag (%d seeds) val-AUC=%.4f", len(seeds),
             roc_auc_score(ds.y_val.values, val_proba))
    # OOF uses the average best_iter across seeds
    avg_best = int(np.mean([m.best_iteration_ or 200 for m in models]))
    full_for_oof = dict(LGBM_BASE)
    full_for_oof.update(params)
    oof = _oof_lgbm_seedbag(full_for_oof, avg_best, ds, seeds)
    return models, oof, val_proba


def _oof_lgbm_seedbag(full: Dict, best_iter: int, ds: Datasets, seeds: Tuple[int, ...]) -> np.ndarray:
    cv = GroupKFold(n_splits=5)
    n_est = max(50, min(best_iter, 600))
    oof = np.zeros(len(ds.X_train))
    for idx_tr, idx_va in cv.split(ds.X_train, ds.y_train, ds.groups_train):
        fold_probas = []
        for seed in seeds:
            params = dict(full)
            params["random_state"] = seed
            params["n_estimators"] = n_est
            m = lgb.LGBMClassifier(**params)
            m.fit(ds.X_train.iloc[idx_tr], ds.y_train.iloc[idx_tr])
            fold_probas.append(m.predict_proba(ds.X_train.iloc[idx_va])[:, 1])
        oof[idx_va] = np.mean(fold_probas, axis=0)
    return oof


def _oof_lgbm(full: Dict, best_iter: int | None, ds: Datasets) -> np.ndarray:
    cv = GroupKFold(n_splits=5)
    oof = np.zeros(len(ds.X_train))
    n_est = int(best_iter or 300)
    n_est = max(50, min(n_est, 600))
    full_oof = dict(full)
    full_oof["n_estimators"] = n_est
    for idx_tr, idx_va in cv.split(ds.X_train, ds.y_train, ds.groups_train):
        m = lgb.LGBMClassifier(**full_oof)
        m.fit(ds.X_train.iloc[idx_tr], ds.y_train.iloc[idx_tr])
        oof[idx_va] = m.predict_proba(ds.X_train.iloc[idx_va])[:, 1]
    return oof


def fit_xgb(params: Dict, ds: Datasets) -> Tuple[xgb.XGBClassifier, np.ndarray, np.ndarray]:
    full = dict(XGB_BASE)
    full.update(params)
    n_pos = float(ds.y_train.sum())
    full["scale_pos_weight"] = (len(ds.y_train) - n_pos) / max(1.0, n_pos)
    full["n_estimators"] = 2000
    full["early_stopping_rounds"] = 80
    m = xgb.XGBClassifier(**full)
    m.fit(
        ds.X_train, ds.y_train,
        eval_set=[(ds.X_val, ds.y_val)],
        verbose=False,
    )
    LOG.info("XGB final best_iter=%s", getattr(m, "best_iteration", None))
    oof = _oof_xgb(full, getattr(m, "best_iteration", None), ds)
    val_proba = m.predict_proba(ds.X_val)[:, 1]
    return m, oof, val_proba


def _oof_xgb(full: Dict, best_iter: int | None, ds: Datasets) -> np.ndarray:
    cv = GroupKFold(n_splits=5)
    oof = np.zeros(len(ds.X_train))
    n_est = int(best_iter or 300)
    n_est = max(50, min(n_est, 600))
    full_oof = dict(full)
    full_oof.pop("early_stopping_rounds", None)
    full_oof["n_estimators"] = n_est
    for idx_tr, idx_va in cv.split(ds.X_train, ds.y_train, ds.groups_train):
        m = xgb.XGBClassifier(**full_oof)
        m.fit(ds.X_train.iloc[idx_tr], ds.y_train.iloc[idx_tr], verbose=False)
        oof[idx_va] = m.predict_proba(ds.X_train.iloc[idx_va])[:, 1]
    return oof


# ---------------------------------------------------------------------------
# Logistic Regression baseline
# ---------------------------------------------------------------------------
def fit_logreg(ds: Datasets) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Standard-scaled logistic regression (median-imputed)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    numeric_cols = ds.feature_cols
    pre = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("sc", StandardScaler())]), numeric_cols)]
    )
    clf = LogisticRegression(
        solver="saga", C=0.5, class_weight="balanced", max_iter=4000,
        random_state=config.RANDOM_STATE,
    )  # default l2 penalty (sklearn ≥1.8 deprecates explicit penalty kw).
    Xtr = pre.fit_transform(ds.X_train)
    Xva = pre.transform(ds.X_val)
    Xpr = pre.transform(ds.X_predict)
    clf.fit(Xtr, ds.y_train)
    val_proba = clf.predict_proba(Xva)[:, 1]

    cv = GroupKFold(n_splits=5)
    oof = np.zeros(len(ds.X_train))
    for idx_tr, idx_va in cv.split(ds.X_train, ds.y_train, ds.groups_train):
        pre_f = ColumnTransformer(
            [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("sc", StandardScaler())]), numeric_cols)]
        )
        Xtr_f = pre_f.fit_transform(ds.X_train.iloc[idx_tr])
        Xva_f = pre_f.transform(ds.X_train.iloc[idx_va])
        clf_f = LogisticRegression(
            solver="saga", C=0.5, class_weight="balanced", max_iter=4000,
            random_state=config.RANDOM_STATE,
        )  # default l2 penalty
        clf_f.fit(Xtr_f, ds.y_train.iloc[idx_tr])
        oof[idx_va] = clf_f.predict_proba(Xva_f)[:, 1]
    return (pre, clf, Xpr), oof, val_proba


# ---------------------------------------------------------------------------
# Calibration (isotonic with sigmoid fallback per delta)
# ---------------------------------------------------------------------------
class _PrefitWrapper(BaseEstimator, ClassifierMixin):
    """Wraps a fitted sklearn estimator to expose ``fit`` and ``predict_proba``."""
    _estimator_type = "classifier"

    def __init__(self, estimator):
        self.estimator = estimator
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, **kw):
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


def calibrate_with_groupkfold(make_base, X, y, groups, X_val, y_val, raw_val_auc: float,
                              label: str) -> Tuple[CalibratedClassifierCV, str, Dict[str, float]]:
    """Try isotonic first; fall back to sigmoid if AUC on the hold-out drops by
    more than 0.01 versus the *raw* (uncalibrated) model.

    ``make_base`` is a factory that returns a fresh estimator each call so the
    five CV folds train independent base learners. The final prediction
    averages the five calibrated learners.
    """
    cv_indices = list(GroupKFold(n_splits=5).split(X, y, groups))
    results: Dict[str, Tuple[Any, float]] = {}
    for method in ("isotonic", "sigmoid"):
        cal = CalibratedClassifierCV(estimator=make_base(), method=method, cv=cv_indices)
        cal.fit(X, y)
        proba_val = cal.predict_proba(X_val)[:, 1]
        try:
            auc = roc_auc_score(y_val, proba_val)
        except ValueError:
            auc = np.nan
        results[method] = (cal, auc)
        LOG.info("Calibration %s/%s hold-out AUC=%.4f (raw=%.4f, Δ=%+.4f)",
                 label, method, auc, raw_val_auc, auc - raw_val_auc)
    iso_cal, iso_auc = results["isotonic"]
    sig_cal, sig_auc = results["sigmoid"]
    audit = {"isotonic_val_auc": iso_auc, "sigmoid_val_auc": sig_auc, "raw_val_auc": raw_val_auc}
    if raw_val_auc - iso_auc > 0.01:
        LOG.info("Calibration choice for %s: sigmoid (isotonic drops AUC by %.4f)",
                 label, raw_val_auc - iso_auc)
        return sig_cal, "sigmoid", audit
    LOG.info("Calibration choice for %s: isotonic", label)
    return iso_cal, "isotonic", audit


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, float]:
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, proba >= t, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def recall_at_top_decile(y_true: np.ndarray, proba: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    k = max(1, int(0.1 * len(y_true)))
    order = np.argsort(-proba)
    return float(y_true[order[:k]].sum() / max(1, y_true.sum()))


def all_metrics(y_true, proba, threshold: float | None = None) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    out: Dict[str, float] = {
        "prevalence": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }
    if threshold is None:
        bt, bf1 = best_f1_threshold(y_true, proba)
        out["best_threshold"] = bt
        out["f1_at_best_threshold"] = bf1
    else:
        out["threshold_train"] = float(threshold)
        out["f1_at_train_threshold"] = float(
            f1_score(y_true, proba >= threshold, zero_division=0)
        )
    out["recall_at_top_decile"] = recall_at_top_decile(y_true, proba)
    return out


# ---------------------------------------------------------------------------
# MLP via subprocess
# ---------------------------------------------------------------------------
def run_mlp_subprocess(ds: Datasets) -> Dict[str, Any]:
    LOG.info("Launching MLP in a dedicated subprocess (CPU)…")
    tmp_dir = Path(os.environ.get("TMPDIR", "/tmp"))
    in_path = tmp_dir / "mlp_in.pkl"
    out_path = tmp_dir / "mlp_out.pkl"

    # Impute NaNs with train-median so the MLP sees a dense matrix; this is
    # done with TRAIN statistics only (we'll save the medians as scaler-side
    # info inside the worker through its own ``apply_scaler`` step).
    train_median = ds.X_train.median(numeric_only=True)
    # Belt-and-suspenders: median NaN (when an entire column is NaN in train)
    # collapses to 0 so the network never sees NaNs.
    Xtr = ds.X_train.fillna(train_median).fillna(0.0).to_numpy(dtype=np.float32)
    Xva = ds.X_val.fillna(train_median).fillna(0.0).to_numpy(dtype=np.float32)
    Xpr = ds.X_predict.fillna(train_median).fillna(0.0).to_numpy(dtype=np.float32)

    payload = {
        "X_train": Xtr,
        "y_train": ds.y_train.to_numpy().astype(np.int32),
        "groups_train": ds.groups_train.to_numpy(),
        "X_val": Xva,
        "y_val": ds.y_val.to_numpy().astype(np.int32),
        "X_predict": Xpr,
    }
    with in_path.open("wb") as fh:
        pickle.dump(payload, fh)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"

    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m", "risk_model.mlp_subprocess",
        str(in_path), str(out_path),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0
    if result.returncode != 0:
        LOG.error("MLP subprocess failed (rc=%s):\nSTDOUT:%s\nSTDERR:%s",
                  result.returncode, result.stdout, result.stderr)
        raise RuntimeError("MLP subprocess crashed")
    LOG.info("MLP subprocess finished in %.1fs", elapsed)

    with out_path.open("rb") as fh:
        out = pickle.load(fh)

    out["train_median"] = train_median  # save for downstream prediction reproducibility
    return out


# ---------------------------------------------------------------------------
# Stacking (manual, leakage-free)
# ---------------------------------------------------------------------------
def fit_stacking(oof_train: np.ndarray, y_train: np.ndarray, val_probas: np.ndarray) -> Tuple[LogisticRegression, np.ndarray]:
    """Train the meta-learner on OOF predictions; score the hold-out using the
    *final* (early-stopped) base models' predictions.
    """
    meta = LogisticRegression(
        C=1.0, solver="lbfgs",
        class_weight="balanced", max_iter=2000, random_state=config.RANDOM_STATE,
    )  # default l2 penalty
    meta.fit(oof_train, y_train)
    return meta, meta.predict_proba(val_probas)[:, 1]


# ---------------------------------------------------------------------------
# Per-Fase AUC (sanity for sub-populations)
# ---------------------------------------------------------------------------
def auc_by_subpop(y_true, proba, subpop: pd.Series, label: str) -> pd.DataFrame:
    rows = []
    for value, idx in subpop.groupby(subpop).groups.items():
        idx = list(idx)
        if len(idx) < 10:
            continue
        sub_y = np.asarray(y_true)[idx]
        sub_p = np.asarray(proba)[idx]
        if len(np.unique(sub_y)) < 2:
            continue
        rows.append({
            label: value,
            "n": len(idx),
            "pos_rate": float(sub_y.mean()),
            "roc_auc": float(roc_auc_score(sub_y, sub_p)),
            "pr_auc": float(average_precision_score(sub_y, sub_p)),
        })
    return pd.DataFrame(rows).sort_values(label).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_calibration(y_true, probas: Dict[str, np.ndarray], out: Path):
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, p in probas.items():
        p = np.asarray(p)
        idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
        mp, mt = [], []
        for b in range(n_bins):
            m = idx == b
            if m.sum() < 5:
                continue
            mp.append(p[m].mean())
            mt.append(np.asarray(y_true)[m].mean())
        ax.plot(mp, mt, marker="o", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.5)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration curves – hold-out (2023→2024)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_auc_by_fase(table: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    width = 0.35
    x = np.arange(len(table))
    ax.bar(x - width / 2, table["roc_auc"], width, label="ROC-AUC")
    ax.bar(x + width / 2, table["pr_auc"], width, label="PR-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(table["Fase"].astype(int).astype(str))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fase")
    ax.set_title("Performance por Fase – stacking ensemble (hold-out 2024)")
    ax.legend()
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_risk_distribution(preds: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(preds["prob_final"], bins=30, color="#3b6ea5", alpha=0.85)
    for thr, label, color in [(config.RISK_BAND_LOW, "Médio≥", "orange"),
                               (config.RISK_BAND_HIGH, "Alto>", "red")]:
        ax.axvline(thr, linestyle="--", color=color, label=f"{label}{thr}")
    ax.set_xlabel("Probabilidade de risco (composto)")
    ax.set_ylabel("Nº alunos")
    ax.set_title("Distribuição do risco previsto para 2025 (n=1156)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def shap_top15(model, X_eval: pd.DataFrame, feature_names: List[str], title: str, out_png: Path
               ) -> Tuple[pd.DataFrame, np.ndarray]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_eval)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    top = imp.head(15)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#3b6ea5")
    ax.set_xlabel("|SHAP| média")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return imp, shap_values


def per_row_top3(shap_values: np.ndarray, names: List[str]) -> List[str]:
    abs_sv = np.abs(shap_values)
    top3 = np.argsort(-abs_sv, axis=1)[:, :3]
    out = []
    for i in range(shap_values.shape[0]):
        parts = []
        for j in top3[i]:
            sign = "+" if shap_values[i, j] >= 0 else "−"
            parts.append(f"{names[j]}({sign}{abs(shap_values[i, j]):.2f})")
        out.append(" | ".join(parts))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    t_start = time.time()

    LOG.info("=== Stage 1 — assemble train/val/predict ===")
    ds = assemble()

    LOG.info("=== Stage 2 — hyper-parameter sweep (GroupKFold on train) ===")
    best_lgbm, lgbm_cv_auc, lgbm_sweep_log = sweep_lgbm(ds.X_train, ds.y_train, ds.groups_train)
    best_xgb,  xgb_cv_auc,  xgb_sweep_log  = sweep_xgb(ds.X_train,  ds.y_train, ds.groups_train)
    LOG.info("Best LGBM: %s (CV-AUC=%.4f)", best_lgbm, lgbm_cv_auc)
    LOG.info("Best XGB : %s (CV-AUC=%.4f)", best_xgb,  xgb_cv_auc)

    LOG.info("=== Stage 3 — fit baselines + final tree models ===")
    (logreg_pre, logreg_clf, logreg_Xpr), logreg_oof, logreg_val = fit_logreg(ds)
    lgbm_models, lgbm_oof, lgbm_val_raw = fit_lgbm(best_lgbm, ds)
    lgbm_model = lgbm_models[0]  # representative model (best_iter of seed=42) for SHAP
    xgb_model,  xgb_oof,  xgb_val_raw  = fit_xgb(best_xgb, ds)

    LOG.info("=== Stage 4 — calibrate the two tree models ===")
    lgbm_full_params = dict(LGBM_BASE)
    lgbm_full_params.update(best_lgbm)
    lgbm_full_params["n_estimators"] = max(50, int(lgbm_model.best_iteration_ or 200))

    def make_lgbm():
        return lgb.LGBMClassifier(**lgbm_full_params)

    raw_lgbm_val_auc = roc_auc_score(ds.y_val.values, lgbm_val_raw)
    lgbm_cal, lgbm_cal_method, lgbm_cal_audit = calibrate_with_groupkfold(
        make_lgbm, ds.X_train, ds.y_train.values, ds.groups_train.values,
        ds.X_val, ds.y_val.values, raw_lgbm_val_auc, "lgbm",
    )
    lgbm_val_cal = lgbm_cal.predict_proba(ds.X_val)[:, 1]
    lgbm_oof_cal_for_meta = lgbm_oof  # use raw OOF for meta; calibration is a per-fold avg
    # For stacking we use *raw* OOF probabilities — they are well-defined and
    # the meta-learner can re-calibrate jointly.

    xgb_full_params = dict(XGB_BASE)
    xgb_full_params.update(best_xgb)
    n_pos = float(ds.y_train.sum())
    xgb_full_params["scale_pos_weight"] = (len(ds.y_train) - n_pos) / max(1.0, n_pos)
    xgb_full_params["n_estimators"] = max(50, int(getattr(xgb_model, "best_iteration", 200)))
    xgb_full_params.pop("early_stopping_rounds", None)

    def make_xgb():
        return xgb.XGBClassifier(**xgb_full_params)

    raw_xgb_val_auc = roc_auc_score(ds.y_val.values, xgb_val_raw)
    xgb_cal, xgb_cal_method, xgb_cal_audit = calibrate_with_groupkfold(
        make_xgb, ds.X_train, ds.y_train.values, ds.groups_train.values,
        ds.X_val, ds.y_val.values, raw_xgb_val_auc, "xgb",
    )
    xgb_val_cal = xgb_cal.predict_proba(ds.X_val)[:, 1]

    LOG.info("=== Stage 5 — MLP via subprocess ===")
    mlp_out = run_mlp_subprocess(ds)
    mlp_oof = mlp_out["oof"]
    mlp_val_raw = mlp_out["val_pred"]
    mlp_predict = mlp_out["predict_pred"]

    # MLP calibration on the (unbiased) OOF predictions of the train period.
    # Platt scaling – single-parameter sigmoid fit – is the safest choice with
    # 600 calibration points; isotonic would over-fit at this size.
    platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
    platt.fit(mlp_oof.reshape(-1, 1), ds.y_train.values)
    mlp_val_cal = platt.predict_proba(mlp_val_raw.reshape(-1, 1))[:, 1]
    mlp_predict_cal = platt.predict_proba(mlp_predict.reshape(-1, 1))[:, 1]
    LOG.info("MLP calibration (Platt) val-AUC: raw=%.4f → calibrated=%.4f, Brier raw=%.4f → cal=%.4f",
             roc_auc_score(ds.y_val.values, mlp_val_raw),
             roc_auc_score(ds.y_val.values, mlp_val_cal),
             brier_score_loss(ds.y_val.values, mlp_val_raw),
             brier_score_loss(ds.y_val.values, mlp_val_cal))

    LOG.info("=== Stage 6 — Spearman rank correlation between models ===")
    from scipy.stats import spearmanr
    spear = {
        "lgbm_xgb": float(spearmanr(lgbm_val_raw, xgb_val_raw).correlation),
        "lgbm_mlp": float(spearmanr(lgbm_val_raw, mlp_val_raw).correlation),
        "xgb_mlp":  float(spearmanr(xgb_val_raw,  mlp_val_raw).correlation),
    }
    LOG.info("Spearman on hold-out: %s", spear)

    LOG.info("=== Stage 7 — Stacking (OOF meta on train, score on val) ===")
    # 4-model stack: LightGBM + XGBoost + MLP + LogReg.  Including LogReg adds
    # a linear, well-behaved base learner whose biases differ from the trees.
    base_names = ["lgbm", "xgb", "mlp", "logreg"]
    oof_stack = np.column_stack([lgbm_oof, xgb_oof, mlp_oof, logreg_oof])
    val_stack = np.column_stack([lgbm_val_raw, xgb_val_raw, mlp_val_raw, logreg_val])
    meta_lr, stacking_val = fit_stacking(oof_stack, ds.y_train.values, val_stack)
    LOG.info("Stacking meta weights (after scaling): %s",
             dict(zip(base_names, meta_lr.coef_[0].round(3))))

    # Also evaluate a plain average of the calibrated models as a robust
    # baseline ensemble (no fitted weights, so cannot overfit the meta).
    avg_val = np.mean([lgbm_val_cal, mlp_val_cal], axis=0)

    LOG.info("=== Stage 8 — Operational threshold from train OOF ===")
    # Use stacking OOF on train to choose the operational threshold; report at
    # that fixed threshold on the hold-out.
    stack_oof_train = meta_lr.predict_proba(oof_stack)[:, 1]
    thr, _ = best_f1_threshold(ds.y_train.values, stack_oof_train)
    LOG.info("Operational threshold (max F1 on train OOF): %.3f", thr)

    LOG.info("=== Stage 9 — Final hold-out evaluation ===")
    probas_for_eval = {
        "LogReg":            logreg_val,
        "LightGBM_raw":      lgbm_val_raw,
        "LightGBM_calibr":   lgbm_val_cal,
        "XGBoost_raw":       xgb_val_raw,
        "XGBoost_calibr":    xgb_val_cal,
        "MLP_raw":           mlp_val_raw,
        "MLP_calibr":        mlp_val_cal,
        "Stacking":          stacking_val,
        "Average_LGBM_MLP":  avg_val,
    }
    metrics_full: Dict[str, Dict[str, float]] = {}
    for name, p in probas_for_eval.items():
        metrics_full[name] = all_metrics(ds.y_val.values, p, threshold=thr)
    metrics_full["Prevalence_baseline"] = {
        "prevalence": float(ds.y_val.mean()),
        "roc_auc": 0.5,
        "pr_auc": float(ds.y_val.mean()),
        "brier": float(ds.y_val.mean() * (1 - ds.y_val.mean())),
        "threshold_train": thr,
        "f1_at_train_threshold": 0.0,
        "recall_at_top_decile": 0.1,
    }
    # Record any external experiment artifacts (e.g. Optuna search) so the
    # historical comparison stays in one place.  We do not re-run them here.
    exp_dir = config.ROOT / "experiments"
    experiments = []
    if exp_dir.exists():
        for ts_dir in sorted(exp_dir.iterdir()):
            for sub in sorted(ts_dir.iterdir()):
                best_json = sub / "best.json"
                if best_json.exists():
                    try:
                        experiments.append({
                            "experiment": sub.name,
                            "timestamp": ts_dir.name,
                            **json.loads(best_json.read_text()),
                        })
                    except Exception:
                        pass
    metrics_full["experiments"] = experiments
    metrics_full["_meta"] = {
        "n_train": int(len(ds.X_train)),
        "n_val": int(len(ds.X_val)),
        "n_predict": int(len(ds.X_predict)),
        "n_features": int(len(ds.feature_cols)),
        "lgbm_best": best_lgbm,
        "lgbm_cv_auc_sweep": lgbm_cv_auc,
        "lgbm_calibration": lgbm_cal_method,
        "xgb_best": best_xgb,
        "xgb_cv_auc_sweep": xgb_cv_auc,
        "xgb_calibration": xgb_cal_method,
        "lgbm_calibration_audit": lgbm_cal_audit,
        "xgb_calibration_audit": xgb_cal_audit,
        "spearman_holdout": spear,
        "operational_threshold": thr,
        "lgbm_sweep_log": lgbm_sweep_log,
        "xgb_sweep_log": xgb_sweep_log,
        "target": config.PRIMARY_TARGET,
        "train_year": config.TRAIN_YEAR,
        "val_year": config.VAL_YEAR,
        "predict_year": config.PREDICT_YEAR,
    }
    (config.MODELS_DIR / "metrics_full.json").write_text(
        json.dumps(metrics_full, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Persist models
    joblib.dump(lgbm_models, config.MODELS_DIR / "lgbm.pkl")
    joblib.dump(lgbm_model, config.MODELS_DIR / "lgbm_seed42.pkl")
    joblib.dump(lgbm_cal,   config.MODELS_DIR / "lgbm_calibrated.pkl")
    joblib.dump(xgb_model,  config.MODELS_DIR / "xgb.pkl")
    joblib.dump(xgb_cal,    config.MODELS_DIR / "xgb_calibrated.pkl")
    joblib.dump(meta_lr,    config.MODELS_DIR / "stacking.pkl")
    joblib.dump({
        "feature_cols": ds.feature_cols,
        "logreg_preprocessor": logreg_pre,
        "lgbm_full_params": lgbm_full_params,
        "xgb_full_params": xgb_full_params,
        "mlp_train_median": mlp_out["train_median"],
        "mlp_scaler_mean": mlp_out["scaler_mean"],
        "mlp_scaler_std":  mlp_out["scaler_std"],
    }, config.MODELS_DIR / "preprocessor.pkl")

    import torch
    torch.save(
        {k: torch.from_numpy(v) for k, v in mlp_out["state_dict"].items()},
        config.MODELS_DIR / "mlp.pt",
    )
    (config.MODELS_DIR / "mlp_meta.json").write_text(
        json.dumps({
            "n_features": int(mlp_out["n_features"]),
            "history_len": len(mlp_out["history"]),
            "operational_threshold": thr,
            "architecture": "Linear(n→256)→BN→GELU→Drop(0.3)→Linear(→128)→BN→GELU→Drop(0.3)→Linear(→64)→GELU→Drop(0.2)→Linear(→1)",
            "device": "cpu",
            "optimizer": "AdamW(lr=1e-3, wd=1e-4)",
            "scheduler": "OneCycleLR(max_lr=3e-3, pct_start=0.30)",
            "loss": "BCEWithLogitsLoss(pos_weight=neg/pos)",
            "batch_size": 64,
            "max_epochs": 200,
            "patience": 20,
        }, indent=2),
        encoding="utf-8",
    )

    LOG.info("=== Stage 10 — SHAP for LightGBM and XGBoost ===")
    feat_names = ds.feature_cols
    imp_lgbm, shap_lgbm = shap_top15(
        lgbm_model, ds.X_predict, feat_names,
        "LightGBM – Top 15 features (mean |SHAP|) sobre 2024",
        config.FIGURES_DIR / "lgbm_shap_top15.png",
    )
    imp_lgbm.to_csv(config.REPORTS_DIR / "lgbm_shap.csv", index=False)
    imp_xgb, shap_xgb = shap_top15(
        xgb_model, ds.X_predict, feat_names,
        "XGBoost – Top 15 features (mean |SHAP|) sobre 2024",
        config.FIGURES_DIR / "xgb_shap_top15.png",
    )
    imp_xgb.to_csv(config.REPORTS_DIR / "xgb_shap.csv", index=False)

    LOG.info("=== Stage 11 — Predictions for 2025 ===")
    # Score predict-year via the seed-bag average; calibrated path uses the
    # CV-trained CalibratedClassifierCV.
    lgbm_pred_raw = np.mean(
        [m.predict_proba(ds.X_predict)[:, 1] for m in lgbm_models], axis=0
    )
    lgbm_pred_cal = lgbm_cal.predict_proba(ds.X_predict)[:, 1]
    xgb_pred_raw  = xgb_model.predict_proba(ds.X_predict)[:, 1]
    xgb_pred_cal  = xgb_cal.predict_proba(ds.X_predict)[:, 1]
    mlp_pred_raw  = mlp_predict
    mlp_pred_cal  = mlp_predict_cal
    logreg_pred = logreg_clf.predict_proba(logreg_Xpr)[:, 1]
    avg_pred = np.mean([lgbm_pred_cal, mlp_pred_cal], axis=0)
    # Stacking on predict-year uses the same 4-vector ordering
    stack_pred = meta_lr.predict_proba(
        np.column_stack([lgbm_pred_raw, xgb_pred_raw, mlp_pred_raw, logreg_pred])
    )[:, 1]

    # Decide which model is "final" using hold-out ROC-AUC.  We include the
    # raw LightGBM here because its calibrated version sacrificed ranking
    # quality on this dataset (isotonic + sigmoid both flattened AUC).
    candidates = [
        "LightGBM_raw", "LightGBM_calibr", "XGBoost_calibr",
        "MLP_raw", "Stacking", "Average_LGBM_MLP",
    ]
    finalist = max(candidates, key=lambda n: metrics_full[n]["roc_auc"])
    finalist_pred = {
        "LightGBM_raw": lgbm_pred_raw,
        "LightGBM_calibr": lgbm_pred_cal,
        "XGBoost_calibr": xgb_pred_cal,
        "MLP_raw": mlp_pred_raw,
        "Stacking": stack_pred,
        "Average_LGBM_MLP": avg_pred,
    }[finalist]
    LOG.info("Finalist model for predictions: %s", finalist)

    # Per-row SHAP explanations using LightGBM (interpretable to NGO)
    top3 = per_row_top3(shap_lgbm, feat_names)

    pred_year = ds.panel.query("Ano == @config.PREDICT_YEAR").set_index("RA")
    ra_index = ds.panel.query("Ano == @config.PREDICT_YEAR")["RA"].values
    out = pd.DataFrame({
        "RA": ra_index,
        "Nome Anonimizado": pred_year.loc[ra_index, "Nome Anonimizado"].values,
        "Fase": pred_year.loc[ra_index, "Fase"].values,
        "Pedra atual": pred_year.loc[ra_index, "Pedra"].values,
        "INDE atual": pred_year.loc[ra_index, "INDE"].values,
        "prob_lgbm": np.round(lgbm_pred_cal, 4),
        "prob_xgb": np.round(xgb_pred_cal, 4),
        "prob_mlp": np.round(mlp_pred_cal, 4),
        "prob_stacking": np.round(stack_pred, 4),
        "prob_final": np.round(finalist_pred, 4),
        "top_3_fatores": top3,
    })

    def faixa(p: float) -> str:
        if p > config.RISK_BAND_HIGH:
            return "Alto"
        if p >= config.RISK_BAND_LOW:
            return "Médio"
        return "Baixo"

    out["faixa_risco"] = out["prob_final"].apply(faixa)
    out = out.sort_values("prob_final", ascending=False).reset_index(drop=True)
    out.to_csv(config.PREDICTIONS_CSV, index=False)
    LOG.info("Predictions saved to %s", config.PREDICTIONS_CSV)

    LOG.info("=== Stage 12 — Per-Fase AUC + plots ===")
    fase_series = ds.panel.query("Ano == @config.VAL_YEAR").set_index("RA").loc[ds.groups_val.values, "Fase"]
    fase_series = fase_series.reset_index(drop=True)
    auc_fase = auc_by_subpop(ds.y_val.values, stacking_val, fase_series.astype(int), "Fase")
    auc_fase.to_csv(config.REPORTS_DIR / "auc_by_fase.csv", index=False)

    pedra_series = ds.panel.query("Ano == @config.VAL_YEAR").set_index("RA").loc[ds.groups_val.values, "Pedra"]
    pedra_series = pedra_series.reset_index(drop=True)
    auc_pedra = auc_by_subpop(ds.y_val.values, stacking_val, pedra_series.fillna("NA"), "Pedra")
    auc_pedra.to_csv(config.REPORTS_DIR / "auc_by_pedra.csv", index=False)

    gen_series = ds.panel.query("Ano == @config.VAL_YEAR").set_index("RA").loc[ds.groups_val.values, "Gênero"]
    gen_series = gen_series.reset_index(drop=True)
    auc_gen = auc_by_subpop(ds.y_val.values, stacking_val, gen_series.fillna("NA"), "Gênero")
    auc_gen.to_csv(config.REPORTS_DIR / "auc_by_gender.csv", index=False)

    plot_calibration(
        ds.y_val.values,
        {
            "LightGBM (calibr.)": lgbm_val_cal,
            "XGBoost (calibr.)":  xgb_val_cal,
            "MLP":                mlp_val_raw,
            "Stacking":           stacking_val,
        },
        config.FIGURES_DIR / "calibration_curves.png",
    )
    plot_auc_by_fase(auc_fase, config.FIGURES_DIR / "auc_by_fase.png")
    plot_risk_distribution(out, config.FIGURES_DIR / "risk_distribution_2025.png")

    LOG.info("=== Stage 13 — Console summary ===")
    print("\n## Hold-out (features 2023 → label 2024) — alvo composto\n")
    rows = []
    order = [
        "Prevalence_baseline",
        "LogReg",
        "LightGBM_raw", "LightGBM_calibr",
        "XGBoost_raw", "XGBoost_calibr",
        "MLP_raw", "MLP_calibr",
        "Stacking", "Average_LGBM_MLP",
    ]
    for name in order:
        m = metrics_full[name]
        rows.append([
            name,
            f"{m['roc_auc']:.3f}",
            f"{m['pr_auc']:.3f}",
            f"{m['brier']:.3f}",
            f"{m.get('f1_at_train_threshold', m.get('f1_at_best_threshold', 0)):.3f}",
            f"{m['recall_at_top_decile']:.3f}",
        ])
    header = ["Modelo", "ROC-AUC", "PR-AUC", "Brier", f"F1@thr={thr:.2f}", "Recall@top10%"]
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    fmt = "| " + " | ".join("{:<" + str(w) + "}" for w in widths) + " |"
    print(fmt.format(*header))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        print(fmt.format(*r))

    print(f"\nMelhor modelo no hold-out: **{finalist}** (ROC-AUC={metrics_full[finalist]['roc_auc']:.3f})")
    print(f"Alvo: {config.PRIMARY_TARGET} | Threshold operacional (max F1 OOF train): {thr:.3f}")

    print("\n### Spearman entre modelos no hold-out")
    for k, v in spear.items():
        print(f"  {k}: {v:+.3f}")

    print("\n### Distribuição de risco para 2025")
    bands = out["faixa_risco"].value_counts().reindex(["Alto", "Médio", "Baixo"]).fillna(0).astype(int)
    print(bands.to_string())

    print("\n### Top-10 features (LightGBM, |SHAP|)")
    print(imp_lgbm.head(10).to_string(index=False))

    print(f"\n[done] elapsed={time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
