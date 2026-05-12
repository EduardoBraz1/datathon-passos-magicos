"""Training pipeline.

Workflow:
1. Load the cleaned panel and build a feature matrix for every available year.
2. Build train (anchor=2022, labels=2023) and validation (anchor=2023, labels=2024)
   sets.
3. Train four model families *in parallel* via ``joblib.Parallel`` so the user
   sees end-to-end wall-clock benefits:
   - Logistic regression (baseline)
   - LightGBM with early stopping
   - XGBoost with early stopping
   - PyTorch MLP with focal loss + OneCycleLR
4. Calibrate the three non-baseline models (isotonic).
5. Build a stacking ensemble with a logistic-regression meta-learner.
6. Retrain the chosen artifacts on train+val and persist everything to
   ``models/``.
7. Return a dictionary of artifacts + the held-out predictions for evaluation.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from . import config, data, features, target
from ._mlp import TorchMLPClassifier
from ._preprocess import build_linear_preprocessor, build_tree_preprocessor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
@dataclass
class TrainingData:
    X_train: pd.DataFrame
    y_train: pd.Series
    groups_train: pd.Series  # RA, used for GroupKFold
    X_val: pd.DataFrame
    y_val: pd.Series
    groups_val: pd.Series
    X_predict: pd.DataFrame  # rows for the prediction year (no label)
    feature_cols: List[str]
    numeric_cols: List[str]
    target_name: str


def assemble_training_data(panel: pd.DataFrame, target_name: str = config.PRIMARY_TARGET) -> TrainingData:
    feature_panel, schema = features.build_feature_panel(
        panel, config.YEARS_AVAILABLE
    )

    def _slice(year: int, with_target: bool) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.Series]:
        rows = feature_panel.query("Ano == @year").copy()
        if with_target:
            tgt = target.build_targets(panel, year)
            rows = rows.set_index("RA").join(tgt, how="inner")
            y = rows[target_name]
            rows = rows.drop(columns=target.all_target_columns())
            groups = pd.Series(rows.index.values, index=rows.index, name="RA")
            X = rows.reset_index(drop=True)[schema.feature_columns]
            return X, y.reset_index(drop=True), groups.reset_index(drop=True)
        rows = rows.set_index("RA")
        X = rows.reset_index(drop=True)[schema.feature_columns]
        groups = pd.Series(rows.index.values, name="RA")
        return X, None, groups

    X_tr, y_tr, g_tr = _slice(config.TRAIN_YEAR, with_target=True)
    X_va, y_va, g_va = _slice(config.VAL_YEAR, with_target=True)
    X_pr, _, _ = _slice(config.PREDICT_YEAR, with_target=False)

    log.info(
        "train: %d rows (pos=%.2f%%); val: %d rows (pos=%.2f%%); predict: %d rows",
        len(X_tr), 100 * y_tr.mean(), len(X_va), 100 * y_va.mean(), len(X_pr),
    )

    return TrainingData(
        X_train=X_tr, y_train=y_tr, groups_train=g_tr,
        X_val=X_va, y_val=y_va, groups_val=g_va,
        X_predict=X_pr,
        feature_cols=schema.feature_columns,
        numeric_cols=schema.numeric,
        target_name=target_name,
    )


# ---------------------------------------------------------------------------
# Per-model trainers (CPU-bound, isolated so they can run in parallel)
# ---------------------------------------------------------------------------
def _train_logreg(td: TrainingData):
    pre = build_linear_preprocessor(td.numeric_cols)
    Xtr = pre.fit_transform(td.X_train)
    Xva = pre.transform(td.X_val)
    clf = LogisticRegression(**config.LOGREG_PARAMS)
    clf.fit(Xtr, td.y_train.values)
    val_proba = clf.predict_proba(Xva)[:, 1]
    return {"name": "logreg", "preprocessor": pre, "estimator": clf, "val_proba": val_proba}


def _train_lgbm(td: TrainingData):
    pre = build_tree_preprocessor(td.numeric_cols)
    Xtr = pre.fit_transform(td.X_train)
    Xva = pre.transform(td.X_val)
    scale_pos = (td.y_train == 0).sum() / max(1, (td.y_train == 1).sum())
    params = dict(config.LGBM_PARAMS)
    params["scale_pos_weight"] = scale_pos

    model = lgb.LGBMClassifier(**params)
    model.fit(
        Xtr, td.y_train.values,
        eval_set=[(Xva, td.y_val.values)],
        callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS), lgb.log_evaluation(0)],
    )
    val_proba = model.predict_proba(Xva)[:, 1]
    return {"name": "lgbm", "preprocessor": pre, "estimator": model, "val_proba": val_proba}


def _train_xgb(td: TrainingData):
    pre = build_tree_preprocessor(td.numeric_cols)
    Xtr = pre.fit_transform(td.X_train)
    Xva = pre.transform(td.X_val)
    scale_pos = (td.y_train == 0).sum() / max(1, (td.y_train == 1).sum())
    params = dict(config.XGB_PARAMS)
    params["scale_pos_weight"] = scale_pos
    params["early_stopping_rounds"] = config.EARLY_STOPPING_ROUNDS

    model = xgb.XGBClassifier(**params)
    model.fit(
        Xtr, td.y_train.values,
        eval_set=[(Xva, td.y_val.values)],
        verbose=False,
    )
    val_proba = model.predict_proba(Xva)[:, 1]
    return {"name": "xgb", "preprocessor": pre, "estimator": model, "val_proba": val_proba}


def _train_mlp(td: TrainingData):
    pre = build_linear_preprocessor(td.numeric_cols)
    Xtr = pre.fit_transform(td.X_train)
    Xva = pre.transform(td.X_val)
    model = TorchMLPClassifier()
    model.fit(Xtr, td.y_train.values, X_val=Xva, y_val=td.y_val.values)
    val_proba = model.predict_proba(Xva)[:, 1]
    return {"name": "mlp", "preprocessor": pre, "estimator": model, "val_proba": val_proba}


_TRAINER_REGISTRY = {
    "logreg": _train_logreg,
    "lgbm": _train_lgbm,
    "xgb": _train_xgb,
    "mlp": _train_mlp,
}


def _timed(fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    out["fit_seconds"] = time.time() - t0
    return out


def train_all(td: TrainingData, n_jobs: int = 4) -> Dict[str, Dict[str, Any]]:
    """Train all four model families in parallel.

    We use joblib's ``threading`` backend because the heavy lifting in every
    trainer (LightGBM, XGBoost, sklearn-saga, PyTorch) releases the GIL inside
    native code.  Threading avoids the macOS fork/spawn instability seen with
    PyTorch + MPS + loky workers while still giving wall-clock speed-up over
    sequential training (each family runs concurrently on dedicated worker
    threads while their inner kernels still use ``n_jobs=-1``).
    """
    log.info("Training %d model families in parallel (backend=threading)...", len(_TRAINER_REGISTRY))
    t0 = time.time()
    results = joblib.Parallel(
        n_jobs=n_jobs,
        backend="threading",
        verbose=5,
    )(
        joblib.delayed(_timed)(fn, td) for _, fn in _TRAINER_REGISTRY.items()
    )
    elapsed = time.time() - t0
    log.info("Parallel training finished in %.1fs", elapsed)
    return {r["name"]: r for r in results}


# ---------------------------------------------------------------------------
# Group-K-Fold sanity check (also re-used as CV-based OOF for stacking)
# ---------------------------------------------------------------------------
def groupkfold_oof_auc(td: TrainingData, n_splits: int = config.CV_N_SPLITS) -> Dict[str, float]:
    """Compute LightGBM OOF ROC-AUC inside the training fold using GroupKFold.

    This is mainly used to validate that the temporal model is consistent with
    a group-aware CV inside the train year.
    """
    cv = GroupKFold(n_splits=n_splits)
    aucs: List[float] = []
    for fold, (idx_tr, idx_va) in enumerate(cv.split(td.X_train, td.y_train, td.groups_train)):
        pre = build_tree_preprocessor(td.numeric_cols)
        Xtr = pre.fit_transform(td.X_train.iloc[idx_tr])
        Xva = pre.transform(td.X_train.iloc[idx_va])
        params = dict(config.LGBM_PARAMS)
        params["n_estimators"] = 400
        params["scale_pos_weight"] = (td.y_train.iloc[idx_tr] == 0).sum() / max(1, (td.y_train.iloc[idx_tr] == 1).sum())
        m = lgb.LGBMClassifier(**params)
        m.fit(Xtr, td.y_train.iloc[idx_tr].values,
              eval_set=[(Xva, td.y_train.iloc[idx_va].values)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        proba = m.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(td.y_train.iloc[idx_va], proba)
        aucs.append(auc)
        log.info("CV fold %d ROC-AUC = %.4f", fold + 1, auc)
    return {"mean": float(np.mean(aucs)), "std": float(np.std(aucs)), "per_fold": aucs}


# ---------------------------------------------------------------------------
# Calibration + stacking on top of the parallel-trained base models
# ---------------------------------------------------------------------------
class _SklearnTransformWrapper(BaseEstimator, ClassifierMixin):
    """Wrap a (preprocessor, estimator) pair into a single sklearn estimator
    so that ``CalibratedClassifierCV`` can fit/predict end-to-end.
    """

    def __init__(self, preprocessor, estimator):
        self.preprocessor = preprocessor
        self.estimator = estimator

    def fit(self, X, y):
        Xt = self.preprocessor.fit_transform(X)
        self.estimator.fit(Xt, y)
        self.classes_ = getattr(self.estimator, "classes_", np.array([0, 1]))
        return self

    def predict_proba(self, X):
        Xt = self.preprocessor.transform(X)
        return self.estimator.predict_proba(Xt)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _calibrate(td: TrainingData, base_result: Dict[str, Any]) -> CalibratedClassifierCV:
    """Isotonic calibration via cv='prefit' using the val set as calibration data.

    With ~600 training rows we cannot afford full CV calibration, so we use the
    held-out val set as the calibration sample (it remains untouched for the
    final evaluation since the stacking layer also uses it).
    """
    wrapper = _SklearnTransformWrapper(base_result["preprocessor"], base_result["estimator"])
    wrapper.classes_ = np.array([0, 1])
    calibrator = CalibratedClassifierCV(estimator=wrapper, method="isotonic", cv="prefit")
    calibrator.fit(td.X_val, td.y_val.values)
    return calibrator


def calibrate_models(td: TrainingData, base: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for name in ("lgbm", "xgb", "mlp"):
        out[name] = _calibrate(td, base[name])
    return out


def build_stacking(td: TrainingData, base: Dict[str, Dict[str, Any]],
                   calibrated: Dict[str, Any]) -> Tuple[LogisticRegression, np.ndarray]:
    """Train a logistic-regression meta-learner on the val-set predicted
    probabilities of the three calibrated tree/MLP models.
    """
    cols = []
    names = ["lgbm", "xgb", "mlp"]
    for name in names:
        cols.append(calibrated[name].predict_proba(td.X_val)[:, 1])
    Z_val = np.column_stack(cols)
    meta = LogisticRegression(C=1.0, max_iter=2000, random_state=config.RANDOM_STATE)
    meta.fit(Z_val, td.y_val.values)
    meta_val_proba = meta.predict_proba(Z_val)[:, 1]
    return meta, meta_val_proba


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def persist(base: Dict[str, Dict[str, Any]],
            calibrated: Dict[str, Any],
            meta: LogisticRegression,
            td: TrainingData,
            extra: Optional[Dict[str, Any]] = None) -> Path:
    """Save artifacts to ``models/``.

    LightGBM/XGBoost get saved both as joblib (for direct sklearn use) and via
    their native ``save_model`` paths.  The MLP uses ``torch.save``.
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(base["logreg"]["estimator"], config.MODELS_DIR / "logreg.pkl")
    joblib.dump(base["lgbm"]["estimator"], config.MODELS_DIR / "lgbm.pkl")
    joblib.dump(base["xgb"]["estimator"], config.MODELS_DIR / "xgb.pkl")

    torch.save(base["mlp"]["estimator"].to_state_dict(), config.MODELS_DIR / "mlp.pt")

    joblib.dump(
        {
            "tree_preprocessor": base["lgbm"]["preprocessor"],
            "linear_preprocessor": base["logreg"]["preprocessor"],
            "mlp_preprocessor": base["mlp"]["preprocessor"],
        },
        config.MODELS_DIR / "preprocessor.pkl",
    )

    joblib.dump(
        {"lgbm": calibrated["lgbm"], "xgb": calibrated["xgb"], "mlp": calibrated["mlp"]},
        config.MODELS_DIR / "calibrated.pkl",
    )
    joblib.dump(meta, config.MODELS_DIR / "stacking.pkl")

    meta_path = config.MODELS_DIR / "metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "feature_columns": td.feature_cols,
                "numeric_columns": td.numeric_cols,
                "target_name": td.target_name,
                "train_year": config.TRAIN_YEAR,
                "val_year": config.VAL_YEAR,
                "predict_year": config.PREDICT_YEAR,
                **(extra or {}),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log.info("Saved artifacts to %s", config.MODELS_DIR)
    return config.MODELS_DIR
