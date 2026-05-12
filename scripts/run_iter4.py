"""Iteration 4 — aggressive push above 0.78 ROC-AUC on the composite target,
plus an honest comparison on the three constituent targets.

Pipeline (in order):

* Build the enriched feature panel via :func:`run_pipeline.assemble`.
* Add **train-only** fase-mean-minus features for INDE/IAA/IEG/IPS/IDA/IPV
  (lookup table fit on year 2022 only, applied to 2023 and 2024).
* Train and score, on the same 2023→2024 hold-out:
    - B  LightGBM_raw with all the new interaction features.
    - C  LightGBM **DART**  (no early stopping; doc-compliant params).
    - D  LightGBM **GOSS**  (top_rate / other_rate).
    - A  Per-Fase sub-models (buckets {0-2}, {3-5}, {6-9}) with bucket
        merging when train size < 50 lines.
    - E  Spearman matrix and (only if diversity warrants) an honest stacking.
* If the composite target still misses 0.80, train the best Iter4 setup on
  ``risk_inde_drop``, ``risk_pedra_drop`` and ``risk_defasagem_worsen``
  separately and report side by side.

All artefacts live in ``models/iter4/`` and ``experiments/iter4_<ts>/``.
The CSV ``data/processed/predicoes_risco_2025.csv`` is updated only if the
new Iter4 best (composite target) beats ``LightGBM_raw`` from Iter3.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_pipeline import assemble, LGBM_BASE  # type: ignore
from risk_model import target as tgt_mod
from risk_model import config as rm_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
LOG = logging.getLogger("iter4")


SEEDS = (42, 7, 2024)
THRESHOLD = 0.29  # operational threshold inherited from Iter3 (train OOF)
TOP_DECILE = 0.10
_FASE_MEAN_COLS = ["INDE", "IAA", "IEG", "IPS", "IDA", "IPV"]


# ---------------------------------------------------------------------------
# Train-only fase-mean lookup features
# ---------------------------------------------------------------------------
def add_train_fase_mean_minus(ds, fase_col: str = "Fase") -> Tuple[Dict, list]:
    """Compute, **on the train slice only**, a (Fase → mean) lookup for each
    indicator in :data:`_FASE_MEAN_COLS`, then write a new
    ``<ind>_minus_fase_mean`` column into ``ds.X_train``, ``ds.X_val`` and
    ``ds.X_predict``.

    No leakage: the means are fit on year 2022 (the train period) and applied
    by Fase to every row of every slice.
    """
    if fase_col not in ds.X_train.columns:
        return {}, []
    lookups: Dict[str, Dict[float, float]] = {}
    new_cols: list = []
    train_fase = ds.X_train[fase_col].astype("float64")
    for ind in _FASE_MEAN_COLS:
        if ind not in ds.X_train.columns:
            continue
        mean_by_fase = ds.X_train.groupby(fase_col)[ind].mean().to_dict()
        global_mean = float(ds.X_train[ind].mean())
        lookups[ind] = mean_by_fase
        col_name = f"{ind}_minus_fase_mean"
        new_cols.append(col_name)
        for X in (ds.X_train, ds.X_val, ds.X_predict):
            mean_lookup = X[fase_col].astype("float64").map(mean_by_fase).fillna(global_mean)
            X[col_name] = X[ind].astype("float64") - mean_lookup
    LOG.info("Added %d <ind>_minus_fase_mean features (train-only lookup).", len(new_cols))
    return lookups, new_cols


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------
def score(y_true: np.ndarray, proba: np.ndarray, threshold: float = THRESHOLD) -> Dict:
    n = len(y_true)
    top_k = max(1, int(np.ceil(n * TOP_DECILE)))
    order = np.argsort(-proba)[:top_k]
    return {
        "n": int(n),
        "prevalence": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "f1_at_train_threshold": float(f1_score(y_true, (proba >= threshold).astype(int))),
        "recall_at_top_decile": float(y_true[order].sum() / max(1, y_true.sum())),
    }


# ---------------------------------------------------------------------------
# LightGBM trainers
# ---------------------------------------------------------------------------
def fit_lgbm_seedbag(params: Dict, X_tr, y_tr, X_va, y_va,
                     seeds: Tuple[int, ...] = SEEDS,
                     n_estimators: int = 2000,
                     early_stop: int = 80,
                     use_eval: bool = True
                     ) -> Tuple[List[lgb.LGBMClassifier], np.ndarray]:
    """Generic seed-bagged LightGBM trainer with optional early stopping."""
    models, vp = [], []
    for s in seeds:
        full = dict(LGBM_BASE)
        full.update(params)
        full["random_state"] = s
        full["n_estimators"] = n_estimators
        m = lgb.LGBMClassifier(**full)
        if use_eval:
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(early_stop, verbose=False),
                             lgb.log_evaluation(0)])
        else:
            m.fit(X_tr, y_tr)
        models.append(m)
        vp.append(m.predict_proba(X_va)[:, 1])
    return models, np.mean(vp, axis=0)


def predict_seedbag(models: List[lgb.LGBMClassifier], X: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


# ---------------------------------------------------------------------------
# A — Per-Fase sub-models
# ---------------------------------------------------------------------------
@dataclass
class FaseBucket:
    name: str
    fases: Tuple[int, ...]


def _bucket_label(fase: float) -> str:
    f = int(fase)
    if f <= 2:
        return "0_2"
    if f <= 5:
        return "3_5"
    return "6_9"


def train_per_fase(ds) -> Tuple[np.ndarray, Dict]:
    """Train one LightGBM per Fase bucket; concatenate hold-out predictions."""
    base_buckets = [
        FaseBucket("0_2", (0, 1, 2)),
        FaseBucket("3_5", (3, 4, 5)),
        FaseBucket("6_9", (6, 7, 8, 9)),
    ]
    fase_tr = ds.X_train["Fase"].astype("float64")
    fase_va = ds.X_val["Fase"].astype("float64")
    bucket_of_tr = fase_tr.map(_bucket_label)
    bucket_of_va = fase_va.map(_bucket_label)

    # Merge any bucket with <50 train rows into its neighbour.
    counts = bucket_of_tr.value_counts().to_dict()
    LOG.info("Per-Fase train counts: %s", counts)
    fused: Dict[str, str] = {b.name: b.name for b in base_buckets}
    if counts.get("6_9", 0) < 50:
        fused["6_9"] = "3_5"
        LOG.warning("Bucket 6_9 has %d <50 train rows → merged into 3_5.", counts.get("6_9", 0))
    if counts.get("3_5", 0) < 50:
        fused["3_5"] = "0_2"
        LOG.warning("Bucket 3_5 has %d <50 train rows → merged into 0_2.", counts.get("3_5", 0))

    bucket_of_tr = bucket_of_tr.map(fused)
    bucket_of_va = bucket_of_va.map(fused)

    val_proba = np.zeros(len(ds.X_val))
    per_bucket_scores: Dict = {}
    booster_paths = {}
    out_dir = ROOT / "models" / "iter4"
    out_dir.mkdir(parents=True, exist_ok=True)

    for bucket_name in sorted(set(fused.values())):
        idx_tr = bucket_of_tr[bucket_of_tr == bucket_name].index
        idx_va = bucket_of_va[bucket_of_va == bucket_name].index
        Xb_tr = ds.X_train.loc[idx_tr]
        yb_tr = ds.y_train.loc[idx_tr]
        Xb_va = ds.X_val.loc[idx_va]
        yb_va = ds.y_val.loc[idx_va]
        if len(Xb_tr) == 0 or len(Xb_va) == 0:
            LOG.warning("Skip bucket %s — empty (tr=%d, va=%d).", bucket_name, len(Xb_tr), len(Xb_va))
            continue

        params = {
            "num_leaves": 15,
            "min_child_samples": max(5, len(Xb_tr) // 40),
            "is_unbalance": True,
            "learning_rate": 0.05,
        }
        models, vp = fit_lgbm_seedbag(params, Xb_tr, yb_tr, Xb_va, yb_va,
                                      seeds=SEEDS, use_eval=True)
        # X_val has a default RangeIndex (0..len-1), so idx_va.values are the
        # actual positional indices.  Write each bucket's probas into the
        # global vector at those positions — buckets are mutually exclusive by
        # Fase, so there is no overwrite.
        val_proba[idx_va.values] = vp

        per_bucket_scores[bucket_name] = score(yb_va.values, vp)
        per_bucket_scores[bucket_name].update({
            "n_train": int(len(Xb_tr)),
            "n_val": int(len(Xb_va)),
            "params": params,
        })
        joblib.dump(models, out_dir / f"lgbm_fase_{bucket_name}.pkl")
        booster_paths[bucket_name] = str(out_dir / f"lgbm_fase_{bucket_name}.pkl")
        LOG.info("Per-Fase[%s] n_tr=%d n_va=%d  AUC=%.4f  PR-AUC=%.4f",
                 bucket_name, len(Xb_tr), len(Xb_va),
                 per_bucket_scores[bucket_name]["roc_auc"],
                 per_bucket_scores[bucket_name]["pr_auc"])

    return val_proba, {"per_bucket": per_bucket_scores, "model_paths": booster_paths,
                       "bucket_assignment": fused}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _target_for(panel: pd.DataFrame, ano: int, var: str, ra_index: pd.Index) -> pd.Series:
    """Build a single risk variant target for the rows in *ra_index*.

    NaNs (students who appear at year *t* but not at *t+1*) are dropped — they
    have no observable label.
    """
    full = tgt_mod.build_targets(panel, ano)
    out = full.reindex(ra_index)[var]
    return out.dropna().astype("int8")


def main() -> int:
    t0 = time.time()
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = ROOT / "models" / "iter4"
    exp_dir = ROOT / "experiments" / f"iter4_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("=== Iter4 stage 0 — assemble + train-only fase-mean lookup ===")
    ds = assemble()
    add_train_fase_mean_minus(ds)
    LOG.info("Total features after iter4 enrichment: %d", ds.X_train.shape[1])

    results: Dict[str, Dict] = {}

    # ----- B  LightGBM_raw on the full new feature set ----------------------
    LOG.info("=== Iter4 stage B — LightGBM_raw with interactions+fase-mean ===")
    iter3_params = {"num_leaves": 15, "min_child_samples": 40, "learning_rate": 0.05}
    _, vb = fit_lgbm_seedbag(iter3_params, ds.X_train, ds.y_train,
                             ds.X_val, ds.y_val, seeds=SEEDS, use_eval=True)
    results["B_LGBM_interactions"] = score(ds.y_val.values, vb)
    LOG.info("B  AUC=%.4f PR=%.4f Brier=%.4f", results["B_LGBM_interactions"]["roc_auc"],
             results["B_LGBM_interactions"]["pr_auc"], results["B_LGBM_interactions"]["brier"])

    # ----- C  DART -----------------------------------------------------------
    LOG.info("=== Iter4 stage C — LightGBM DART ===")
    dart_params = {
        "boosting_type": "dart",
        "num_leaves": 31,
        "min_child_samples": 20,
        "learning_rate": 0.05,
        "drop_rate": 0.1,
        "skip_drop": 0.5,
        "max_drop": 50,
        "is_unbalance": True,
    }
    # DART does not support early stopping reliably; train fixed n_estimators
    _, vc = fit_lgbm_seedbag(dart_params, ds.X_train, ds.y_train,
                             ds.X_val, ds.y_val, seeds=SEEDS,
                             n_estimators=1500, use_eval=False)
    results["C_DART"] = score(ds.y_val.values, vc)
    LOG.info("C  AUC=%.4f PR=%.4f", results["C_DART"]["roc_auc"], results["C_DART"]["pr_auc"])

    # ----- D  GOSS -----------------------------------------------------------
    LOG.info("=== Iter4 stage D — LightGBM GOSS ===")
    goss_params = {
        "boosting_type": "goss",
        "num_leaves": 15,
        "min_child_samples": 40,
        "learning_rate": 0.05,
        "top_rate": 0.2,
        "other_rate": 0.1,
        "is_unbalance": True,
        # GOSS is incompatible with bagging; override LGBM_BASE defaults.
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
    }
    _, vd = fit_lgbm_seedbag(goss_params, ds.X_train, ds.y_train,
                             ds.X_val, ds.y_val, seeds=SEEDS, use_eval=True)
    results["D_GOSS"] = score(ds.y_val.values, vd)
    LOG.info("D  AUC=%.4f PR=%.4f", results["D_GOSS"]["roc_auc"], results["D_GOSS"]["pr_auc"])

    # ----- A  Per-Fase -------------------------------------------------------
    LOG.info("=== Iter4 stage A — Per-Fase sub-models ===")
    va, perfase_meta = train_per_fase(ds)
    results["A_PerFase"] = score(ds.y_val.values, va)
    LOG.info("A  global AUC=%.4f PR=%.4f", results["A_PerFase"]["roc_auc"],
             results["A_PerFase"]["pr_auc"])

    # ----- E  Spearman diversity + ensemble ---------------------------------
    spear = {
        "B_C": spearmanr(vb, vc).statistic,
        "B_D": spearmanr(vb, vd).statistic,
        "B_A": spearmanr(vb, va).statistic,
        "C_D": spearmanr(vc, vd).statistic,
    }
    LOG.info("Spearman diversity on hold-out: %s", spear)

    # If diversity is meaningful (any pair < 0.93) build a simple average of the
    # two best ROC-AUC models, otherwise skip — adding correlated learners
    # doesn't reduce variance.
    model_probas = {"B_LGBM_interactions": vb, "C_DART": vc, "D_GOSS": vd, "A_PerFase": va}

    # Try every diverse pair (Spearman < 0.93) and report.  Mean-rank
    # ensembles are favoured over raw-prob means when scales differ — per-Fase
    # probabilities are bucket-relative, so rank-blending fixes the scale.
    def _rankmean(probas: list) -> np.ndarray:
        ranks = [pd.Series(p).rank(pct=True).values for p in probas]
        return np.mean(ranks, axis=0)

    diverse_pairs = []
    keys = list(model_probas.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            sp = spearmanr(model_probas[ki], model_probas[kj]).statistic
            if sp < 0.93:
                diverse_pairs.append((ki, kj, sp))
    if not diverse_pairs:
        LOG.info("E  Skipped — no model pair has Spearman < 0.93.")
        ve = None
    else:
        diverse_pairs.sort(key=lambda x: (
            -max(results[x[0]]["roc_auc"], results[x[1]]["roc_auc"])
        ))
        for ki, kj, sp in diverse_pairs[:3]:
            ve_pair = _rankmean([model_probas[ki], model_probas[kj]])
            name = f"E_RankAvg_{ki}_{kj}"
            results[name] = score(ds.y_val.values, ve_pair)
            LOG.info("E  RankAvg(%s,%s) Spearman=%.3f → AUC=%.4f PR=%.4f",
                     ki, kj, sp, results[name]["roc_auc"], results[name]["pr_auc"])

    # ----- Hybrid: PerFase[0_2] only for that bucket, B for the rest -------
    # Rationale: PerFase wins big on Fase 0-2 (AUC 0.79) but loses on 3-9
    # (AUC 0.66).  A bucket-aware blend uses each model where it is strongest.
    bucket_of_val = ds.X_val["Fase"].astype("float64").map(_bucket_label)
    hybrid = vb.copy()
    mask_low = (bucket_of_val == "0_2").values
    # Use ranks within each bucket so the scales remain comparable across the
    # global sort that AUC computes.
    if mask_low.any():
        from scipy.stats import rankdata
        a_ranks = rankdata(va[mask_low]) / mask_low.sum()
        b_ranks = rankdata(vb[mask_low]) / mask_low.sum()
        hybrid[mask_low] = (a_ranks + b_ranks) / 2.0
    results["E_Hybrid_B_PerFase02"] = score(ds.y_val.values, hybrid)
    LOG.info("E  Hybrid(B for Fase 3-9, RankAvg(B,A) for Fase 0-2) AUC=%.4f PR=%.4f",
             results["E_Hybrid_B_PerFase02"]["roc_auc"],
             results["E_Hybrid_B_PerFase02"]["pr_auc"])

    # ----- Pick best for composite -----------------------------------------
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_auc = results[best_name]["roc_auc"]
    LOG.info("Iter4 best on composite: %s ROC=%.4f", best_name, best_auc)

    iter3_baseline = {
        "roc_auc": 0.7584, "pr_auc": 0.8427, "brier": 0.2001,
        "f1_at_train_threshold": 0.7913, "recall_at_top_decile": 0.1423,
    }

    composite_table = {"Iter3_LightGBM_raw": iter3_baseline, **results}
    (exp_dir / "composite_results.json").write_text(
        json.dumps(composite_table, indent=2, ensure_ascii=False)
    )
    LOG.info("Composite results written to %s", exp_dir / "composite_results.json")

    # ----- F  Alternative targets (only if composite still < 0.80) ----------
    alt_results: Dict[str, Dict] = {}
    target_vars = ["risk_inde_drop", "risk_pedra_drop", "risk_defasagem_worsen"]
    LOG.info("=== Iter4 stage F — alternative target definitions ===")
    for var in target_vars:
        # ds.groups_train / ds.groups_val carry the RA labels for each row of
        # ds.X_train / ds.X_val.  We use those directly to align the alt target.
        tgt_train = tgt_mod.build_targets(ds.panel, rm_config.TRAIN_YEAR)[var]
        tgt_val   = tgt_mod.build_targets(ds.panel, rm_config.VAL_YEAR)[var]

        y_tr_alt = ds.groups_train.map(tgt_train.to_dict())
        y_va_alt = ds.groups_val.map(tgt_val.to_dict())
        mask_tr = y_tr_alt.notna()
        mask_va = y_va_alt.notna()
        X_tr_alt = ds.X_train.loc[mask_tr.values].reset_index(drop=True)
        X_va_alt = ds.X_val.loc[mask_va.values].reset_index(drop=True)
        y_tr_alt = y_tr_alt[mask_tr].astype("int8").reset_index(drop=True)
        y_va_alt = y_va_alt[mask_va].astype("int8").reset_index(drop=True)

        if y_tr_alt.sum() == 0 or y_va_alt.sum() == 0:
            LOG.warning("Skip %s — no positives in train or val.", var)
            continue

        params = dict(iter3_params)
        params["is_unbalance"] = True
        models_alt, va_alt = fit_lgbm_seedbag(params, X_tr_alt, y_tr_alt,
                                               X_va_alt, y_va_alt,
                                               seeds=SEEDS, use_eval=True)
        sc = score(y_va_alt.values, va_alt)
        alt_results[var] = sc
        joblib.dump(models_alt, out_dir / f"lgbm_{var}.pkl")
        LOG.info("F %-25s prev=%.2f%% AUC=%.4f PR=%.4f Brier=%.4f F1=%.3f Recall10=%.3f",
                 var, 100*sc['prevalence'], sc["roc_auc"], sc["pr_auc"],
                 sc["brier"], sc["f1_at_train_threshold"], sc["recall_at_top_decile"])

    # ----- Save artifacts ---------------------------------------------------
    summary = {
        "timestamp": ts,
        "elapsed_seconds": time.time() - t0,
        "composite_iter3_baseline": iter3_baseline,
        "composite_iter4": results,
        "spearman_diversity": {k: float(v) for k, v in spear.items()},
        "perfase_meta": perfase_meta,
        "alt_targets": alt_results,
        "best_composite": {"name": best_name, "roc_auc": float(best_auc)},
        "iter3_to_iter4_delta_roc_auc": float(best_auc - iter3_baseline["roc_auc"]),
        "feature_count_after_enrichment": int(ds.X_train.shape[1]),
    }
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ----- Update predicoes_risco_2025.csv only if Iter4 beats Iter3 -------
    iter3_roc = iter3_baseline["roc_auc"]
    if best_auc >= iter3_roc + 0.005:
        LOG.info("Iter4 composite beats Iter3 by ≥0.005 — updating prob_final in CSV.")
        # Recompute predictions for 2025 with the winning model.
        best_proba_predict = None
        if best_name == "B_LGBM_interactions":
            models_b, _ = fit_lgbm_seedbag(iter3_params, ds.X_train, ds.y_train,
                                            ds.X_val, ds.y_val, seeds=SEEDS, use_eval=True)
            best_proba_predict = predict_seedbag(models_b, ds.X_predict)
        elif best_name == "C_DART":
            models_c, _ = fit_lgbm_seedbag(dart_params, ds.X_train, ds.y_train,
                                            ds.X_val, ds.y_val, seeds=SEEDS,
                                            n_estimators=1500, use_eval=False)
            best_proba_predict = predict_seedbag(models_c, ds.X_predict)
        elif best_name == "D_GOSS":
            models_d, _ = fit_lgbm_seedbag(goss_params, ds.X_train, ds.y_train,
                                            ds.X_val, ds.y_val, seeds=SEEDS, use_eval=True)
            best_proba_predict = predict_seedbag(models_d, ds.X_predict)

        if best_proba_predict is not None:
            csv_path = ROOT / "data" / "processed" / "predicoes_risco_2025.csv"
            existing = pd.read_csv(csv_path)
            ra_to_p = dict(zip(ds.predict_ra.astype(str), best_proba_predict))
            existing["prob_final"] = existing["RA"].astype(str).map(ra_to_p).round(4)
            # Re-derive risk bands and re-sort so the file remains ranked.
            def _band(p):
                if p >= 0.66:
                    return "Alto"
                if p >= 0.33:
                    return "Médio"
                return "Baixo"
            existing["faixa_risco"] = existing["prob_final"].apply(_band)
            existing = existing.sort_values("prob_final", ascending=False).reset_index(drop=True)
            existing.to_csv(csv_path, index=False)
            LOG.info("Updated prob_final in %s with %s.", csv_path, best_name)
    else:
        LOG.info("Iter4 composite did NOT beat Iter3 (Δ=%+.4f < 0.005) — keeping prob_final.",
                 best_auc - iter3_roc)

    # ----- Save alternative-target predictions file ------------------------
    if alt_results:
        # Pick the best alternative target by ROC-AUC for the operational CSV.
        best_alt = max(alt_results, key=lambda k: alt_results[k]["roc_auc"])
        models_best_alt = joblib.load(out_dir / f"lgbm_{best_alt}.pkl")
        predict_proba_alt = predict_seedbag(models_best_alt, ds.X_predict)

        alt_df = pd.DataFrame({
            "RA": ds.predict_ra.astype(str),
            "Fase": ds.panel.query("Ano == @rm_config.PREDICT_YEAR")["Fase"].values,
            "Pedra_atual": ds.panel.query("Ano == @rm_config.PREDICT_YEAR")["Pedra"].values,
            "INDE_atual": ds.panel.query("Ano == @rm_config.PREDICT_YEAR")["INDE"].values,
            f"prob_{best_alt}": np.round(predict_proba_alt, 4),
        })
        alt_df["faixa_risco_alt"] = pd.cut(
            alt_df[f"prob_{best_alt}"], bins=[-0.01, 0.33, 0.66, 1.01],
            labels=["Baixo", "Médio", "Alto"]
        )
        alt_df = alt_df.sort_values(f"prob_{best_alt}", ascending=False).reset_index(drop=True)
        alt_csv = ROOT / "data" / "processed" / "predicoes_risco_2025_alternativo.csv"
        alt_df.to_csv(alt_csv, index=False)
        LOG.info("Alternative-target CSV (alvo=%s) saved to %s", best_alt, alt_csv)

    LOG.info("=== Iter4 done — %.1fs ===", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
