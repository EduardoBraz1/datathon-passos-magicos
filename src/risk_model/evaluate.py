"""Evaluation, calibration plots and metric persistence."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from . import config

log = logging.getLogger(__name__)


def _best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, float]:
    """Sweep thresholds and return (best_threshold, best_f1)."""
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(y_true, proba >= t, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def _recall_at_top_decile(y_true: np.ndarray, proba: np.ndarray) -> float:
    n = len(y_true)
    k = max(1, int(0.1 * n))
    order = np.argsort(-proba)
    top = order[:k]
    return float(y_true[top].sum() / max(1, y_true.sum()))


def evaluate_predictions(y_true: Iterable[int], proba: Iterable[float], name: str) -> Dict[str, float]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(proba), dtype=float)
    metrics: Dict[str, float] = {
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }
    best_t, best_f1 = _best_f1_threshold(y, p)
    metrics["best_threshold"] = best_t
    metrics["f1_at_best_threshold"] = best_f1
    metrics["recall_at_top_decile"] = _recall_at_top_decile(y, p)
    log.info(
        "[%s] ROC-AUC=%.3f PR-AUC=%.3f Brier=%.3f F1*=%.3f@%.2f rec@top10%%=%.3f",
        name, metrics["roc_auc"], metrics["pr_auc"], metrics["brier"],
        metrics["f1_at_best_threshold"], metrics["best_threshold"], metrics["recall_at_top_decile"],
    )
    return metrics


def evaluate_models(
    y_true: Iterable[int],
    proba_by_model: Dict[str, np.ndarray],
    save_to: Path | None = None,
) -> Dict[str, Dict[str, float]]:
    results = {name: evaluate_predictions(y_true, p, name) for name, p in proba_by_model.items()}
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Saved metrics to %s", save_to)
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_roc_pr(y_true, proba_by_model: Dict[str, np.ndarray], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, proba in proba_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, proba):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.6)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves – held-out year 2024")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    p = out_dir / "roc_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, proba in proba_by_model.items():
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ax.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_true, proba):.3f})")
    ax.axhline(np.mean(y_true), color="grey", linestyle="--", alpha=0.6, label="prevalence")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves – held-out year 2024")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    p = out_dir / "pr_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    return saved


def plot_calibration(y_true, proba_by_model: Dict[str, np.ndarray], out_dir: Path,
                     n_bins: int = 10) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, n_bins + 1)
    for name, proba in proba_by_model.items():
        prob = np.asarray(proba)
        true = np.asarray(y_true)
        idx = np.digitize(prob, bins) - 1
        idx = np.clip(idx, 0, n_bins - 1)
        mean_pred, mean_true = [], []
        for b in range(n_bins):
            mask = idx == b
            if mask.sum() < 5:
                continue
            mean_pred.append(prob[mask].mean())
            mean_true.append(true[mask].mean())
        ax.plot(mean_pred, mean_true, marker="o", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.6)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration – held-out year 2024")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    p = out_dir / "calibration.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def summary_table(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for name, m in metrics.items():
        rows.append({
            "modelo": name,
            "ROC-AUC": m["roc_auc"],
            "PR-AUC": m["pr_auc"],
            "Brier": m["brier"],
            "F1*": m["f1_at_best_threshold"],
            "Thr*": m["best_threshold"],
            "Recall@top10%": m["recall_at_top_decile"],
        })
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
