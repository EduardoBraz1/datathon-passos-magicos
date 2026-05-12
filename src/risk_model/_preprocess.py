"""Reusable preprocessing pipelines.

Two flavours are produced:

* ``tree_preprocessor``:   one-hot for categoricals, imputation, **no** scaling
  – appropriate for LightGBM / XGBoost which are scale-invariant.
* ``linear_preprocessor``: one-hot + median impute + standard-scale – required
  for Logistic Regression and the PyTorch MLP.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_COLS


def _onehot() -> OneHotEncoder:
    # sklearn ≥1.4: sparse_output kwarg; we want dense for downstream stacking.
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10)


def build_tree_preprocessor(numeric_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", _onehot(), CATEGORICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_linear_preprocessor(numeric_cols: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", _onehot(), CATEGORICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
