"""Target construction for the risk-of-defasagem problem.

The whole modelling exercise hinges on how we define "academic risk".  We
implement four explicit variants and a composite that is the project's PRIMARY
target.  Every target is computed from year *t+1* observations of the student
and aligned with features at year *t* — no future information leaks into the
predictors.
"""
from __future__ import annotations

import logging
from typing import Iterable, List

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


def _pedra_drop(row_t: pd.Series, row_t1: pd.Series) -> bool:
    """True if Pedra at t+1 is strictly worse than at t, or stuck at Quartzo."""
    pt, pt1 = row_t.get("Pedra_ord"), row_t1.get("Pedra_ord")
    if pd.isna(pt) or pd.isna(pt1):
        return False
    if pt1 < pt:
        return True
    return bool(pt == 0 and pt1 == 0)


def _build_pairs(panel: pd.DataFrame, year_t: int) -> pd.DataFrame:
    """Inner-join the (t) and (t+1) snapshots on RA, suffixing columns."""
    year_t1 = year_t + 1
    a = panel.query("Ano == @year_t").set_index("RA")
    b = panel.query("Ano == @year_t1").set_index("RA")
    common = a.index.intersection(b.index)
    a = a.loc[common].add_suffix("_t")
    b = b.loc[common].add_suffix("_t1")
    return a.join(b)


def build_targets(panel: pd.DataFrame, year_t: int) -> pd.DataFrame:
    """For a given feature year *t*, compute all target variants from year *t+1*.

    Returns a DataFrame indexed by RA with one row per supervisable student.
    """
    pairs = _build_pairs(panel, year_t)

    inde_t = pairs["INDE_t"].astype("float64")
    inde_t1 = pairs["INDE_t1"].astype("float64")

    # 1) Pedra drop (strict regression or stuck at Quartzo)
    pedra_drop = pairs.apply(
        lambda r: _pedra_drop(
            r.filter(regex=r"_t$").rename(lambda c: c[:-2]),
            r.filter(regex=r"_t1$").rename(lambda c: c[:-3]),
        ),
        axis=1,
    )
    # Fast equivalent using vectorised ops (validated against the row-wise version):
    pedra_t = pairs["Pedra_ord_t"].astype("float64")
    pedra_t1 = pairs["Pedra_ord_t1"].astype("float64")
    pedra_drop_vec = ((pedra_t1 < pedra_t) | ((pedra_t == 0) & (pedra_t1 == 0))).fillna(False)
    pedra_drop = pedra_drop_vec.astype(bool)

    # 2) INDE drop > threshold OR (optionally) below P25 of the cohort at t+1
    cohort_p25 = inde_t1.quantile(0.25)
    inde_drop_mag = (inde_t1 - inde_t) < -config.INDE_DROP_THRESHOLD
    inde_drop = inde_drop_mag.fillna(False)
    if config.USE_INDE_P25_FLAG:
        below_p25 = (inde_t1 < cohort_p25).fillna(False)
        inde_drop = inde_drop | below_p25

    # 3) Defasagem worsens (gets bigger) OR Defasagem at t+1 is at least +1.
    def_t = pairs["Defasagem_t"].astype("float64")
    def_t1 = pairs["Defasagem_t1"].astype("float64")
    def_worsen = ((def_t1 > def_t) | (def_t1 >= 1)).fillna(False)

    # 4) Composite (union)
    composite = pedra_drop | inde_drop | def_worsen

    out = pd.DataFrame(
        {
            "risk_pedra_drop": pedra_drop.astype("int8"),
            "risk_inde_drop": inde_drop.astype("int8"),
            "risk_defasagem_worsen": def_worsen.astype("int8"),
            "risk_composite": composite.astype("int8"),
        }
    )
    out.index.name = "RA"

    log.info(
        "Targets for t=%d -> t+1=%d on n=%d students: "
        "pedra=%.2f%%, inde=%.2f%%, defasagem=%.2f%%, composite=%.2f%%",
        year_t, year_t + 1, len(out),
        100 * out["risk_pedra_drop"].mean(),
        100 * out["risk_inde_drop"].mean(),
        100 * out["risk_defasagem_worsen"].mean(),
        100 * out["risk_composite"].mean(),
    )
    return out


def all_target_columns() -> List[str]:
    return list(config.TARGET_VARIANTS)
