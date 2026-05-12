"""Central configuration for the risk-modelling pipeline.

All paths, seeds, target definitions, and hyper-parameter grids live here so that
the rest of the code base remains declarative and easy to audit.  A senior data
scientist would gather every "magic number" in a single file – this is that file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
BASE_CSV = DATA_PROCESSED / "base_historico.csv"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = ROOT / "docs"
PREDICTIONS_CSV = DATA_PROCESSED / "predicoes_risco_2025.csv"

for _p in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR, DOCS_DIR, DATA_PROCESSED):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Temporal split (year of features t -> label observed at t+1)
# ---------------------------------------------------------------------------
TRAIN_YEAR: int = 2022           # features from this year; labels from 2023
VAL_YEAR: int = 2023             # features from this year; labels from 2024
PREDICT_YEAR: int = 2024         # features from this year; predict risk for 2025

YEARS_AVAILABLE: List[int] = [2022, 2023, 2024]

# ---------------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------------
PEDRA_ORDER: Dict[str, int] = {
    "Quartzo": 0,
    "Agata": 1,
    "Ametista": 2,
    "Topázio": 3,
}

# INDE drop magnitude that we consider "academic regression" (calibrated against
# the within-cohort standard deviation, ~1.0).
INDE_DROP_THRESHOLD: float = 0.5
# Whether the inde_drop target should also trigger when t+1 INDE falls below the
# 25th-percentile of the cohort at t+1 (captures absolute, not just relative,
# under-performance).
USE_INDE_P25_FLAG: bool = True

TARGET_VARIANTS: List[str] = [
    "risk_pedra_drop",
    "risk_inde_drop",
    "risk_defasagem_worsen",
    "risk_composite",
]
PRIMARY_TARGET: str = "risk_composite"

# Risk-band cut-offs used in the predictions deliverable.
RISK_BAND_LOW: float = 0.30
RISK_BAND_HIGH: float = 0.60

# ---------------------------------------------------------------------------
# Model hyper-parameter grids – kept compact on purpose. With ~1.4k labelled
# rows, very large grids overfit the small validation set; we lean on sensible
# defaults that worked in the offline iteration loop documented in
# docs/MODELO_RISCO.md.
# ---------------------------------------------------------------------------
LGBM_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "learning_rate": 0.04,
    "num_leaves": 31,
    "min_data_in_leaf": 15,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "n_estimators": 1200,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

XGB_PARAMS: Dict[str, Any] = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "logloss",
    "learning_rate": 0.04,
    "max_depth": 5,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "n_estimators": 1200,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

LOGREG_PARAMS: Dict[str, Any] = {
    "solver": "saga",
    "C": 0.5,
    "max_iter": 4000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}


@dataclass
class MLPConfig:
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.30
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 80
    patience: int = 12
    batch_size: int = 64
    focal_gamma: float = 2.0
    one_cycle_pct_start: float = 0.30


MLP_CONFIG = MLPConfig()

CV_N_SPLITS: int = 5
EARLY_STOPPING_ROUNDS: int = 75

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_LEVEL = "INFO"
