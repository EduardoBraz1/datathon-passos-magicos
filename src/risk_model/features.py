"""Feature engineering for the longitudinal risk panel.

The function :func:`build_feature_panel` consumes the cleaned long-format panel
(one row per student-year) and outputs a *modelling matrix* where every row is
"student RA at year *t*" with features that **only** depend on data available
at year *t* or before.  Labels (computed in :mod:`target`) are joined later by
the trainer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


INDICATOR_COLS = ["INDE", "IAA", "IEG", "IPS", "IDA", "Mat", "Por", "Ing", "IPV", "IAN", "IPP"]
SCALAR_NUM_COLS = ["Idade", "Fase", "Ano ingresso", "Pedra_ord", "Defasagem", "FaseIdeal_num", "Nº Av"]
CATEGORICAL_COLS = ["Gênero", "Instituição de ensino"]


@dataclass
class FeatureSchema:
    """Light contract describing the columns produced by :func:`build_feature_panel`."""

    numeric: List[str]
    categorical: List[str]
    feature_columns: List[str]


def _previous_year_features(panel: pd.DataFrame, year_t: int) -> pd.DataFrame:
    """For each (RA, year_t) row, attach the previous-year indicator deltas.

    For students with no year_t-1 observation we **structurally** fill:
      - ``prev_<ind>`` = current-year value (i.e. assume baseline = current),
      - ``delta_<ind>`` = 0 (no observed change),
      - ``had_prev_year`` = 0 (flag the model can use).

    This avoids creating columns that are 100% NaN for the first year of every
    student (which broke median-imputation in early experiments) while keeping
    the missing-ness information explicit.
    """
    a = panel.query("Ano == @year_t").set_index("RA")
    a_prev = panel.query("Ano == @year_t - 1").set_index("RA").reindex(a.index)
    deltas = pd.DataFrame(index=a.index)
    for c in INDICATOR_COLS:
        cur = a[c].astype("float64")
        prev = a_prev[c].astype("float64")
        prev_filled = prev.fillna(cur)
        deltas[f"prev_{c}"] = prev_filled
        deltas[f"delta_{c}"] = (cur - prev_filled).fillna(0.0)
    deltas["had_prev_year"] = a_prev["Ano"].notna().astype("int8")
    return deltas


def _rolling_features(panel: pd.DataFrame, year_t: int) -> pd.DataFrame:
    """Two-year rolling mean and std of each indicator, ending at year_t (inclusive).

    Std collapses to NaN when the window has a single observation; we fill those
    with 0 (no observed variability) so the feature is informative for both
    students with a single year of history and those with two.  Mean NaNs (rare,
    only when both observations are missing) fall back to the current row's
    value.
    """
    a = panel.query("Ano <= @year_t").copy().sort_values(["RA", "Ano"])
    rolled = (
        a.groupby("RA")[INDICATOR_COLS]
        .rolling(window=2, min_periods=1)
        .agg(["mean", "std"])
    )
    rolled.columns = [f"roll2_{ind}_{stat}" for ind, stat in rolled.columns]
    rolled = rolled.reset_index().drop(columns=["level_1"], errors="ignore")
    rolled["__year__"] = a["Ano"].values
    rolled = rolled.query("__year__ == @year_t").drop(columns="__year__").set_index("RA")
    for c in rolled.columns:
        if c.endswith("_std"):
            rolled[c] = rolled[c].fillna(0.0)
    return rolled


_ZSCORE_INDICATORS = ["INDE", "IAA", "IEG", "IPS", "IDA", "IPV"]


_EPS = 1e-6
_RATIO_TOP_FEATURES = ("Defasagem", "Idade", "IPV")


def _interactions(curr: pd.DataFrame) -> pd.DataFrame:
    """Hand-crafted interactions known to be predictive from EDA, plus the
    2nd-order interactions requested in iter4:

    * Ratios between related indicators (with ``*_was_zero`` flag).
    * Quadratic and ``log1p`` transforms of the three top-SHAP features
      (``Defasagem``, ``Idade``, ``IPV``) — sometimes recover monotonicity
      that a shallow GBM otherwise approximates with multiple splits.
    """
    out = pd.DataFrame(index=curr.index)
    out["gap_Mat_Por"] = curr["Mat"] - curr["Por"]
    out["gap_Mat_Ing"] = curr["Mat"] - curr["Ing"]
    out["gap_Por_Ing"] = curr["Por"] - curr["Ing"]
    out["gap_IPS_IDA"] = curr["IPS"] - curr["IDA"]
    out["gap_IAA_IEG"] = curr["IAA"] - curr["IEG"]
    g = curr.groupby("Fase")["INDE"]
    out["inde_z_fase"] = (curr["INDE"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    out["age_fase_excess"] = curr["Idade"] - (curr["Fase"] + 6).astype("float64")

    # Ratios (with epsilon and "was_zero" flag for denominator).
    ratio_pairs = [
        ("IDA", "INDE"), ("IPS", "IAA"), ("Mat", "Por"),
        ("Por", "Ing"), ("IPV", "IEG"),
    ]
    for num, den in ratio_pairs:
        den_vals = curr[den].astype("float64")
        out[f"ratio_{num}_{den}"] = curr[num].astype("float64") / (den_vals.abs() + _EPS)
        out[f"ratio_{num}_{den}_was_zero"] = (den_vals.abs() < _EPS).astype("int8")

    # Quadratic + log1p on top-SHAP features.  log1p uses |x| so negative
    # values (deltas can be negative for log1p inputs we never feed negative
    # bases below — Defasagem ≥ 0, Idade ≥ 6, IPV ∈ [0, 10]).
    for col in _RATIO_TOP_FEATURES:
        if col in curr.columns:
            v = curr[col].astype("float64")
            out[f"{col}_sq"] = v ** 2
            out[f"{col}_log1p"] = np.log1p(v.clip(lower=0))
    return out


def _cohort_zscores(curr: pd.DataFrame, ano: int) -> pd.DataFrame:
    """Z-score and Fase-internal decile of key indicators within the (Fase × Ano)
    cohort of the row.  Only uses observations of year ``ano`` (current row's
    own year) — by construction this is information available at time t.
    """
    out = pd.DataFrame(index=curr.index)
    for col in _ZSCORE_INDICATORS:
        gm = curr.groupby("Fase")[col]
        mean = gm.transform("mean")
        std = gm.transform("std").replace(0, np.nan)
        out[f"{col}_z_faseAno"] = (curr[col] - mean) / std
        # decile inside Fase (10 quantile-rank buckets, NaN-safe)
        ranks = gm.rank(pct=True, method="average")
        out[f"{col}_decile_fase"] = (ranks * 10).fillna(-1).astype("float64")
    out["__cohort_year__"] = ano
    return out.drop(columns="__cohort_year__")


def _trend_slope(panel: pd.DataFrame, year_t: int) -> pd.DataFrame:
    """Linear trend across the student's history up to t (inclusive).

    For students with only one observation, slope = 0 and ``trend_window`` = 1.
    For 2+ observations we compute least-squares slope on (year, value).  The
    sign of the slope already encodes direction of improvement/regression.
    """
    hist = panel.query("Ano <= @year_t").sort_values(["RA", "Ano"]).copy()
    out_index = panel.query("Ano == @year_t").set_index("RA").index
    slopes = pd.DataFrame(index=out_index)
    grouped = hist.groupby("RA", sort=False)
    for col in _ZSCORE_INDICATORS + ["Defasagem"]:
        slopes[f"slope_{col}"] = 0.0
    slopes["trend_window"] = 1

    for ra, sub in grouped:
        if ra not in out_index:
            continue
        if len(sub) < 2:
            continue
        x = sub["Ano"].astype("float64").values
        x_centered = x - x.mean()
        denom = (x_centered ** 2).sum()
        if denom == 0:
            continue
        for col in _ZSCORE_INDICATORS + ["Defasagem"]:
            y = sub[col].astype("float64").values
            if np.isnan(y).all():
                continue
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                continue
            xc = x[mask] - x[mask].mean()
            yc = y[mask]
            d = (xc ** 2).sum()
            if d == 0:
                continue
            slope = (xc * (yc - yc.mean())).sum() / d
            slopes.at[ra, f"slope_{col}"] = slope
        slopes.at[ra, "trend_window"] = int(len(sub))
    return slopes


def _missingness_flags(curr: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=curr.index)
    for c in INDICATOR_COLS:
        flags[f"{c}_was_missing"] = curr[c].isna().astype("int8")
    flags["Pedra_was_missing"] = curr["Pedra_ord"].isna().astype("int8")
    return flags


def _impute_ing_by_fase(curr: pd.DataFrame) -> pd.Series:
    """Impute the (heavily-missing) English score with the median *of the same
    Fase* — global median would bias early Fases that do not study English yet.
    """
    medians = curr.groupby("Fase")["Ing"].transform("median")
    out = curr["Ing"].fillna(medians)
    # Cohorts where every student is missing English collapse to global median.
    out = out.fillna(curr["Ing"].median())
    # Fases that never assess English (0-2, 8-9) end up with all-NaN groups → use 0.
    return out.fillna(0.0)


def _years_in_program(panel: pd.DataFrame, year_t: int) -> pd.Series:
    counts = panel.query("Ano <= @year_t").groupby("RA").size().rename("years_in_program")
    counts = counts.reindex(panel.query("Ano == @year_t")["RA"]).fillna(0).astype("int8")
    counts.index = panel.query("Ano == @year_t")["RA"].values
    return counts


_PEDRA_HISTORY_BASE_YEAR = 2020


def _pedra_history_features(panel: pd.DataFrame, year_t: int) -> pd.DataFrame:
    """Convert the wide Pedra 20/21/22/23 columns merged from the xlsx into:

    * ``pedra_lag<k>_ord``: ordinal Pedra k years before the anchor (k=1..3),
      taken from the appropriate Pedra YY column in the source xlsx.
    * ``pedra_history_depth``: number of past years with a non-null Pedra
      record (proxy for tenure).
    * ``pedra_slope``: simple slope across observed past years (positive ⇒
      ascending trajectory).
    """
    curr = panel.query("Ano == @year_t").set_index("RA").copy()
    out = pd.DataFrame(index=curr.index)
    order_map = {"Quartzo": 0, "Agata": 1, "Ametista": 2, "Topázio": 3}

    # Map the requested lag to the actual column name in the merged panel.
    history_cols = []
    for k in (1, 2, 3, 4):
        yy = year_t - k
        col = f"Pedra_{yy % 100:02d}"
        if col in curr.columns:
            history_cols.append((k, col))

    for k, col in history_cols:
        ord_series = curr[col].map(order_map).astype("Float32")
        out[f"pedra_lag{k}_ord"] = ord_series

    if history_cols:
        # Depth of available Pedra history.
        depth_cols = [out[f"pedra_lag{k}_ord"].notna() for k, _ in history_cols]
        out["pedra_history_depth"] = pd.concat(depth_cols, axis=1).sum(axis=1).astype("int8")

        # Slope of pedra over time (using available lags).
        slopes = []
        for ra in out.index:
            xs, ys = [], []
            for k, _ in history_cols:
                v = out.at[ra, f"pedra_lag{k}_ord"]
                if pd.notna(v):
                    xs.append(-k)  # x is time relative to anchor (earlier=more negative)
                    ys.append(float(v))
            curr_ord = panel.query("Ano == @year_t").set_index("RA").loc[ra, "Pedra_ord"]
            if pd.notna(curr_ord):
                xs.append(0)
                ys.append(float(curr_ord))
            if len(xs) >= 2:
                x = np.asarray(xs, dtype=float)
                y = np.asarray(ys, dtype=float)
                x_c = x - x.mean()
                d = (x_c ** 2).sum()
                slope = 0.0 if d == 0 else float((x_c * (y - y.mean())).sum() / d)
            else:
                slope = 0.0
            slopes.append(slope)
        out["pedra_slope"] = slopes
        # Delta vs current
        if "Pedra_ord" in panel.columns:
            curr_ord = panel.query("Ano == @year_t").set_index("RA")["Pedra_ord"].astype("float64")
            for k, _ in history_cols:
                out[f"pedra_delta_{k}"] = (curr_ord - out[f"pedra_lag{k}_ord"].astype("float64"))
    else:
        out["pedra_history_depth"] = 0
        out["pedra_slope"] = 0.0

    return out


FASE_MEAN_INDICATORS = ("INDE", "IAA", "IEG", "IPS", "IDA", "IPV")


def fit_fase_mean_lookup(X_train: pd.DataFrame,
                         fase_col: str = "Fase",
                         indicators: Tuple[str, ...] = FASE_MEAN_INDICATORS,
                         ) -> Dict[str, Dict]:
    """Fit a (Fase → mean) lookup for each indicator, **only** on the train
    slice.  The returned dict is stable and serialisable; it is later applied
    via :func:`apply_fase_mean_lookup` to train / val / predict slices.
    """
    lookup: Dict[str, Dict] = {}
    for ind in indicators:
        if ind not in X_train.columns:
            continue
        mean_by_fase = X_train.groupby(fase_col)[ind].mean().to_dict()
        global_mean = float(X_train[ind].mean())
        lookup[ind] = {"per_fase": mean_by_fase, "global": global_mean}
    return lookup


def apply_fase_mean_lookup(X: pd.DataFrame, lookup: Dict[str, Dict],
                           fase_col: str = "Fase") -> List[str]:
    """In-place add ``<ind>_minus_fase_mean`` columns to *X* using the fitted
    lookup.  Returns the list of new column names for downstream tracking.
    """
    new_cols: List[str] = []
    if fase_col not in X.columns:
        return new_cols
    fase = X[fase_col].astype("float64")
    for ind, table in lookup.items():
        if ind not in X.columns:
            continue
        mean_lookup = fase.map(table["per_fase"]).fillna(table["global"])
        col = f"{ind}_minus_fase_mean"
        X[col] = X[ind].astype("float64") - mean_lookup
        new_cols.append(col)
    return new_cols


def _is_new_student(panel: pd.DataFrame, year_t: int) -> pd.Series:
    """1 if this is the first year of the student in the program."""
    curr = panel.query("Ano == @year_t").set_index("RA").index
    prev = panel.query("Ano < @year_t").set_index("RA").index
    return pd.Series(
        [int(ra not in prev) for ra in curr],
        index=curr,
        name="is_new_student",
        dtype="int8",
    )


def build_year_features(panel: pd.DataFrame, year_t: int) -> pd.DataFrame:
    """Build the feature matrix for the rows whose anchor year is *year_t*.

    Every column is computed from observations at year ``year_t`` or earlier —
    no t+1 information leaks into the modelling matrix.
    """
    curr = panel.query("Ano == @year_t").set_index("RA").copy()

    curr["Ing"] = _impute_ing_by_fase(curr)

    base = curr[SCALAR_NUM_COLS + INDICATOR_COLS + CATEGORICAL_COLS].copy()

    missing = _missingness_flags(panel.query("Ano == @year_t").set_index("RA"))

    deltas = _previous_year_features(panel, year_t)
    rolling = _rolling_features(panel, year_t)
    interactions = _interactions(curr)
    cohort = _cohort_zscores(curr, year_t)
    slopes = _trend_slope(panel, year_t).reindex(curr.index)

    yip = _years_in_program(panel, year_t).reindex(curr.index)
    new_flag = _is_new_student(panel, year_t).reindex(curr.index)

    # Pedra history (only present if the panel was enriched with the xlsx).
    if any(c.startswith("Pedra_2") for c in panel.columns):
        pedra_hist = _pedra_history_features(panel, year_t)
    else:
        pedra_hist = pd.DataFrame(index=curr.index)

    feats = pd.concat(
        [
            base, deltas, rolling, interactions, cohort, slopes, missing,
            pedra_hist,
            yip.rename("years_in_program"),
            new_flag,
        ],
        axis=1,
    )
    feats["Ano"] = year_t
    feats = feats.reset_index()
    return feats


def _preimpute_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-Fase median imputation of structurally missing indicators applied to
    the entire panel BEFORE we derive lag/rolling features.  This is critical
    for ``Ing``, which is NaN for the early Fases (English is not taught yet);
    derived columns like ``prev_Ing`` and ``roll2_Ing_mean`` would otherwise
    inherit the gaps.
    """
    panel = panel.copy()
    medians = panel.groupby("Fase")["Ing"].transform("median")
    panel["Ing"] = panel["Ing"].fillna(medians).fillna(panel["Ing"].median()).fillna(0.0)
    return panel


def build_feature_panel(panel: pd.DataFrame, years_t: List[int]) -> Tuple[pd.DataFrame, FeatureSchema]:
    """Build the stacked feature matrix for several anchor years."""
    panel = _preimpute_panel(panel)
    frames = [build_year_features(panel, y) for y in years_t]
    full = pd.concat(frames, axis=0, ignore_index=True)

    feature_cols = [c for c in full.columns if c not in {"RA", "Ano"}]
    numeric = [c for c in feature_cols if c not in CATEGORICAL_COLS]

    # Final safety net: drop any feature that is still 100% NaN after all the
    # structural fills above (the model can't learn from it).  Log it loudly so
    # we notice during pipeline runs.
    all_nan = [c for c in numeric if full[c].isna().all()]
    if all_nan:
        log.warning("Dropping %d all-NaN feature columns: %s", len(all_nan), all_nan)
        full = full.drop(columns=all_nan)
        numeric = [c for c in numeric if c not in all_nan]
        feature_cols = [c for c in feature_cols if c not in all_nan]

    nan_share = full[numeric].isna().mean().sort_values(ascending=False)
    high_nan = nan_share[nan_share > 0.5]
    if not high_nan.empty:
        log.warning("Features with >50%% NaN that will be median-imputed:\n%s", high_nan.to_string())

    schema = FeatureSchema(numeric=numeric, categorical=CATEGORICAL_COLS, feature_columns=feature_cols)
    log.info(
        "Feature matrix: %d rows × %d cols (numeric=%d, categorical=%d) for years=%s",
        len(full), len(feature_cols), len(numeric), len(CATEGORICAL_COLS), years_t,
    )
    return full, schema
