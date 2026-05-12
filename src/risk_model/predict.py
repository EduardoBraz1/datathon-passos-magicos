"""Score the prediction year and persist deliverables to disk."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch

from . import config, data, features
from ._mlp import TorchMLPClassifier

log = logging.getLogger(__name__)


def _risk_band(p: float) -> str:
    if p >= config.RISK_BAND_HIGH:
        return "Alto"
    if p >= config.RISK_BAND_LOW:
        return "Médio"
    return "Baixo"


def _load_artifacts():
    meta = json.loads((config.MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    preprocessors = joblib.load(config.MODELS_DIR / "preprocessor.pkl")
    calibrated = joblib.load(config.MODELS_DIR / "calibrated.pkl")
    stacking = joblib.load(config.MODELS_DIR / "stacking.pkl")
    return meta, preprocessors, calibrated, stacking


def _score_with_stack(X_df: pd.DataFrame, calibrated, stacking) -> np.ndarray:
    cols = []
    for name in ("lgbm", "xgb", "mlp"):
        cols.append(calibrated[name].predict_proba(X_df)[:, 1])
    Z = np.column_stack(cols)
    return stacking.predict_proba(Z)[:, 1]


def predict_year(
    panel: Optional[pd.DataFrame] = None,
    contribs: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Score config.PREDICT_YEAR and write the deliverable CSV."""
    output_path = output_path or config.PREDICTIONS_CSV

    panel = panel if panel is not None else data.load_panel()
    feature_panel, _ = features.build_feature_panel(panel, config.YEARS_AVAILABLE)
    pred_rows = feature_panel.query("Ano == @config.PREDICT_YEAR").reset_index(drop=True)

    meta, _preprocessors, calibrated, stacking = _load_artifacts()
    feature_cols: List[str] = meta["feature_columns"]
    X_pred = pred_rows[feature_cols]

    proba = _score_with_stack(X_pred, calibrated, stacking)

    panel_pred_year = panel.query("Ano == @config.PREDICT_YEAR").set_index("RA")
    out = pd.DataFrame(
        {
            "RA": pred_rows["RA"],
            "Nome Anonimizado": panel_pred_year.loc[pred_rows["RA"], "Nome Anonimizado"].values,
            "Fase": panel_pred_year.loc[pred_rows["RA"], "Fase"].values,
            "Pedra atual": panel_pred_year.loc[pred_rows["RA"], "Pedra"].values,
            "INDE atual": panel_pred_year.loc[pred_rows["RA"], "INDE"].values,
            "prob_risco": np.round(proba, 4),
        }
    )
    out["faixa_risco"] = out["prob_risco"].apply(_risk_band)
    if contribs is not None and len(contribs) == len(out):
        out["top_fatores"] = contribs["top_fatores"].values
    else:
        out["top_fatores"] = ""

    out = out.sort_values("prob_risco", ascending=False).reset_index(drop=True)
    out.to_csv(output_path, index=False)
    log.info("Saved predictions to %s (%d students)", output_path, len(out))
    return out
