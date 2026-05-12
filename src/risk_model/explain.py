"""SHAP-based interpretability for the best tree model."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import config

log = logging.getLogger(__name__)


def _feature_names_after_preprocess(preprocessor, X_df: pd.DataFrame) -> List[str]:
    try:
        names = preprocessor.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:  # pragma: no cover - sklearn version safety net
        return [f"f{i}" for i in range(preprocessor.transform(X_df.head(1)).shape[1])]


def explain_tree_model(
    estimator,
    preprocessor,
    X_train_df: pd.DataFrame,
    X_eval_df: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = preprocessor.transform(X_train_df)
    X_eval = preprocessor.transform(X_eval_df)
    names = _feature_names_after_preprocess(preprocessor, X_train_df)

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_eval)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = (
        pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv(out_dir / "shap_feature_importance.csv", index=False)

    top = imp.head(15)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
    ax.set_xlabel("|SHAP| média")
    ax.set_title("Top 15 features – LightGBM (média do |SHAP| sobre o conjunto de validação)")
    fig.tight_layout()
    fig.savefig(out_dir / "shap_top15.png", dpi=150)
    plt.close(fig)

    # Per-row top-3 features (signed contribution).
    abs_sv = np.abs(shap_values)
    top3_idx = np.argsort(-abs_sv, axis=1)[:, :3]
    per_row = []
    for i in range(shap_values.shape[0]):
        cols = []
        for j in top3_idx[i]:
            sign = "+" if shap_values[i, j] >= 0 else "-"
            cols.append(f"{names[j]}({sign}{abs(shap_values[i, j]):.2f})")
        per_row.append(" | ".join(cols))
    contrib_df = pd.DataFrame({"top_fatores": per_row})

    log.info("SHAP artifacts written to %s", out_dir)
    return {"feature_importance": imp, "row_contributions": contrib_df}
