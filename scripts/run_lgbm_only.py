"""Scoped pipeline: LightGBM only.

Run with::

    .venv/bin/python scripts/run_lgbm_only.py

Steps
-----
1. Load + clean the longitudinal panel.
2. Build features and the composite target (t=2022→2023 for training and
   t=2023→2024 for the temporal hold-out).
3. Train one LightGBM classifier with senior defaults and early stopping on
   the hold-out.
4. Calibrate it via 5-fold GroupKFold isotonic regression (with sigmoid fallback).
5. Evaluate ROC-AUC, PR-AUC, Brier, F1@best-threshold, recall@top-decile
   against the prevalence baseline. Persist to ``models/metrics_lgbm.json``.
6. Save the raw booster (``models/lgbm.pkl``) and the calibrated wrapper
   (``models/lgbm_calibrated.pkl``).
7. SHAP TreeExplainer → plot + CSV.
8. Score 2024 features → ``data/processed/predicoes_risco_2025.csv``.
9. Print a compact Markdown summary on stdout.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from risk_model import config, data, features, target  # noqa: E402

LOG = logging.getLogger("lgbm_only")

LGBM_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "n_estimators": 2000,
    "is_unbalance": True,
    "n_jobs": -1,
    "random_state": config.RANDOM_STATE,
    "verbose": -1,
}
EARLY_STOPPING_ROUNDS = 80


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def prepare_datasets() -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series,
                                pd.DataFrame, pd.DataFrame, List[str]]:
    panel = data.load_panel()
    feature_panel, schema = features.build_feature_panel(panel, config.YEARS_AVAILABLE)

    def slice_year(year: int, with_target: bool):
        rows = feature_panel.query("Ano == @year").copy()
        if with_target:
            tgt = target.build_targets(panel, year)
            rows = rows.set_index("RA").join(tgt, how="inner")
            y = rows[config.PRIMARY_TARGET]
            rows = rows.drop(columns=target.all_target_columns())
            groups = pd.Series(rows.index.values, index=rows.index)
            X = rows[schema.feature_columns]
            return X, y, groups
        rows = rows.set_index("RA")
        return rows[schema.feature_columns], None, pd.Series(rows.index.values, index=rows.index)

    X_tr, y_tr, g_tr = slice_year(config.TRAIN_YEAR, with_target=True)
    X_va, y_va, g_va = slice_year(config.VAL_YEAR, with_target=True)
    X_pr, _, _ = slice_year(config.PREDICT_YEAR, with_target=False)

    LOG.info(
        "train n=%d (pos=%.2f%%); val n=%d (pos=%.2f%%); predict n=%d",
        len(X_tr), 100 * y_tr.mean(), len(X_va), 100 * y_va.mean(), len(X_pr),
    )

    return X_tr, y_tr, g_tr, X_va, y_va, g_va, X_pr, panel, schema.feature_columns


# ---------------------------------------------------------------------------
# Pre-processing: lightweight – LightGBM handles raw numerics; we one-hot the
# two categorical columns to keep the booster simple.
# ---------------------------------------------------------------------------
def encode_categoricals(X_tr: pd.DataFrame, X_va: pd.DataFrame, X_pr: pd.DataFrame
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    cat_cols = ["Gênero", "Instituição de ensino"]
    combined = pd.concat([X_tr, X_va, X_pr], keys=["tr", "va", "pr"])
    encoded = pd.get_dummies(combined, columns=cat_cols, dummy_na=False, drop_first=False)
    encoded.columns = [str(c) for c in encoded.columns]
    X_tr2 = encoded.xs("tr").reset_index(drop=True)
    X_va2 = encoded.xs("va").reset_index(drop=True)
    X_pr2 = encoded.xs("pr").reset_index(drop=True)
    return X_tr2, X_va2, X_pr2, list(encoded.columns)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------
def train_lgbm(X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_tr, y_tr.values,
        eval_set=[(X_va, y_va.values)],
        eval_metric=["binary_logloss", "auc"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    LOG.info("LightGBM best iteration: %s", model.best_iteration_)
    return model


def calibrate(model: lgb.LGBMClassifier, X_tr: pd.DataFrame, y_tr: pd.Series, groups: pd.Series
              ) -> CalibratedClassifierCV:
    """5-fold GroupKFold isotonic calibration, with sigmoid fallback if isotonic fails."""
    cv = list(GroupKFold(n_splits=5).split(X_tr, y_tr, groups))
    # Use the pre-fit estimator's best iteration as the base in each fold by
    # cloning with the same hyper-parameters. CalibratedClassifierCV refits
    # internally, but we keep early-stopping rounds off for these short folds.
    base_params = dict(LGBM_PARAMS)
    base_params["n_estimators"] = max(50, model.best_iteration_ or 200)
    base_params.pop("verbose", None)
    base = lgb.LGBMClassifier(**base_params)
    for method in ("isotonic", "sigmoid"):
        try:
            cal = CalibratedClassifierCV(estimator=base, method=method, cv=cv)
            cal.fit(X_tr, y_tr.values)
            LOG.info("Calibration succeeded with method=%s", method)
            return cal
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Calibration with %s failed (%s); trying fallback.", method, exc)
    raise RuntimeError("Both isotonic and sigmoid calibration failed.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _best_f1(y_true, proba) -> Tuple[float, float]:
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, proba >= t, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def _recall_top_decile(y_true, proba) -> float:
    n = len(y_true)
    k = max(1, int(0.1 * n))
    order = np.argsort(-proba)
    return float(np.asarray(y_true)[order[:k]].sum() / max(1, np.asarray(y_true).sum()))


def evaluate(y_true, probas: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    out = {}
    for name, p in probas.items():
        y = np.asarray(y_true)
        p = np.asarray(p)
        thr, f1 = _best_f1(y, p)
        out[name] = {
            "prevalence": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "best_threshold": thr,
            "f1_at_best_threshold": f1,
            "recall_at_top_decile": _recall_top_decile(y, p),
        }
    return out


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def shap_explain(model: lgb.LGBMClassifier, X_eval: pd.DataFrame, feature_names: List[str]
                 ) -> Tuple[pd.DataFrame, np.ndarray]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_eval)
    if isinstance(shap_values, list):  # older shap API
        shap_values = shap_values[1]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return importance, shap_values


def per_row_top3(shap_values: np.ndarray, feature_names: List[str]) -> List[str]:
    abs_sv = np.abs(shap_values)
    top3 = np.argsort(-abs_sv, axis=1)[:, :3]
    out = []
    for i in range(shap_values.shape[0]):
        parts = []
        for j in top3[i]:
            sign = "+" if shap_values[i, j] >= 0 else "−"
            parts.append(f"{feature_names[j]}({sign}{abs(shap_values[i, j]):.2f})")
        out.append(" | ".join(parts))
    return out


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def faixa_risco(p: float) -> str:
    if p > config.RISK_BAND_HIGH:
        return "Alto"
    if p >= config.RISK_BAND_LOW:
        return "Médio"
    return "Baixo"


def build_predictions(panel: pd.DataFrame, X_pr_features: pd.DataFrame,
                      proba_pr: np.ndarray, top3: List[str]) -> pd.DataFrame:
    pred_year = panel.query("Ano == @config.PREDICT_YEAR").set_index("RA")
    ra_index = panel.query("Ano == @config.PREDICT_YEAR")["RA"].values
    out = pd.DataFrame({
        "RA": ra_index,
        "Nome Anonimizado": pred_year.loc[ra_index, "Nome Anonimizado"].values,
        "Fase": pred_year.loc[ra_index, "Fase"].values,
        "Pedra atual": pred_year.loc[ra_index, "Pedra"].values,
        "INDE atual": pred_year.loc[ra_index, "INDE"].values,
        "prob_risco": np.round(proba_pr, 4),
        "top_3_fatores": top3,
    })
    out["faixa_risco"] = out["prob_risco"].apply(faixa_risco)
    out = out.sort_values("prob_risco", ascending=False).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    t0 = time.time()
    LOG.info("=== LightGBM-only pipeline ===")

    X_tr, y_tr, g_tr, X_va, y_va, g_va, X_pr, panel, _ = prepare_datasets()
    X_tr_enc, X_va_enc, X_pr_enc, feature_names = encode_categoricals(X_tr, X_va, X_pr)
    LOG.info("Encoded feature matrix shape: train=%s val=%s predict=%s",
             X_tr_enc.shape, X_va_enc.shape, X_pr_enc.shape)

    LOG.info("Training LightGBM…")
    t_train = time.time()
    model = train_lgbm(X_tr_enc, y_tr, X_va_enc, y_va)
    LOG.info("LightGBM training finished in %.1fs", time.time() - t_train)

    LOG.info("Calibrating with 5-fold GroupKFold (isotonic→sigmoid fallback)…")
    t_cal = time.time()
    calibrated = calibrate(model, X_tr_enc, y_tr, g_tr)
    LOG.info("Calibration finished in %.1fs", time.time() - t_cal)

    proba_va_raw = model.predict_proba(X_va_enc)[:, 1]
    proba_va_cal = calibrated.predict_proba(X_va_enc)[:, 1]
    proba_pr = calibrated.predict_proba(X_pr_enc)[:, 1]

    metrics = evaluate(y_va.values, {
        "lgbm_raw": proba_va_raw,
        "lgbm_calibrated": proba_va_cal,
    })
    metrics["_meta"] = {
        "n_train": int(len(X_tr_enc)),
        "n_val": int(len(X_va_enc)),
        "n_predict": int(len(X_pr_enc)),
        "n_features": int(X_tr_enc.shape[1]),
        "best_iteration": int(model.best_iteration_ or LGBM_PARAMS["n_estimators"]),
        "target": config.PRIMARY_TARGET,
        "train_year": config.TRAIN_YEAR,
        "val_year": config.VAL_YEAR,
        "predict_year": config.PREDICT_YEAR,
        "hyperparameters": LGBM_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    }
    metrics_path = config.MODELS_DIR / "metrics_lgbm.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Metrics written to %s", metrics_path)

    joblib.dump(model, config.MODELS_DIR / "lgbm.pkl")
    joblib.dump(calibrated, config.MODELS_DIR / "lgbm_calibrated.pkl")
    LOG.info("Saved booster + calibrated wrapper to %s", config.MODELS_DIR)

    LOG.info("Computing SHAP values on the 2024 prediction set…")
    importance, shap_values = shap_explain(model, X_pr_enc, feature_names)
    importance.to_csv(config.REPORTS_DIR / "lgbm_shap.csv", index=False)

    top = importance.head(15)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#3b6ea5")
    ax.set_xlabel("|SHAP| média (escala log-odds)")
    ax.set_title("LightGBM – Top 15 features (mean |SHAP|) sobre 2024")
    fig.tight_layout()
    shap_plot_path = config.FIGURES_DIR / "lgbm_shap_top15.png"
    fig.savefig(shap_plot_path, dpi=150)
    plt.close(fig)
    LOG.info("SHAP table → %s ; plot → %s", config.REPORTS_DIR / 'lgbm_shap.csv', shap_plot_path)

    top3 = per_row_top3(shap_values, feature_names)
    preds = build_predictions(panel, X_pr_enc, proba_pr, top3)
    preds.to_csv(config.PREDICTIONS_CSV, index=False)
    LOG.info("Predictions saved to %s", config.PREDICTIONS_CSV)

    print("\n## Hold-out (features 2023 → label 2024)\n")
    rows = []
    for name in ("lgbm_raw", "lgbm_calibrated"):
        m = metrics[name]
        rows.append([
            name,
            f"{m['roc_auc']:.3f}",
            f"{m['pr_auc']:.3f}",
            f"{m['brier']:.3f}",
            f"{m['f1_at_best_threshold']:.3f} @ {m['best_threshold']:.2f}",
            f"{m['recall_at_top_decile']:.3f}",
            f"{m['prevalence']:.3f}",
        ])
    header = ["modelo", "ROC-AUC", "PR-AUC", "Brier", "F1* @ thr", "Recall@top10%", "Prevalência"]
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    fmt = "| " + " | ".join("{:<" + str(w) + "}" for w in widths) + " |"
    print(fmt.format(*header))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        print(fmt.format(*r))

    print("\n## Distribuição de risco para 2025\n")
    bands = preds["faixa_risco"].value_counts().reindex(["Alto", "Médio", "Baixo"]).fillna(0).astype(int)
    print(bands.to_string())

    print("\n## Top-10 features por |SHAP|\n")
    print(importance.head(10).to_string(index=False))

    print(f"\n[done] elapsed={time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
