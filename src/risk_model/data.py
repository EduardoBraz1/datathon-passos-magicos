"""Load and clean the consolidated PEDE longitudinal panel.

The output is a tidy long-format DataFrame indexed by (RA, Ano) with the
columns required for feature engineering downstream.  All cleaning rules are
explicit, justified by the EDA captured in the notebooks, and idempotent.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


# Columns that are numerically interpretable performance indicators.
INDICATOR_COLS = ["INDE", "IAA", "IEG", "IPS", "IDA", "Mat", "Por", "Ing", "IPV", "IAN", "IPP"]
META_COLS = [
    "RA", "Ano", "Fase", "Turma", "Nome Anonimizado", "Idade", "Gênero",
    "Ano ingresso", "Instituição de ensino", "Pedra", "Nº Av",
    "Fase Ideal", "Defasagem", "Escola", "Ativo/ Inativo",
]


def _normalize_gender(s: pd.Series) -> pd.Series:
    """Unify gender labels (Menina/Feminino → Feminino, Menino/Masculino → Masculino)."""
    mapping = {"Menina": "Feminino", "Menino": "Masculino"}
    return s.replace(mapping)


def _normalize_pedra(s: pd.Series) -> pd.Series:
    """Merge "Agata" / "Ágata" duplicate spellings."""
    return s.replace({"Ágata": "Agata"})


def _normalize_instituicao(s: pd.Series) -> pd.Series:
    """Collapse the noisy 'Instituição de ensino' free-text categories.

    The raw column has 12+ variants with typo-level differences. We group them
    into 5 buckets reflecting the meaningful structural difference for the
    model: public school, private with scholarship, private (paying), already
    graduated and unknown.
    """
    s = s.astype(str).str.strip()
    mapping_rules = {
        "Pública": "Publica",
        "Escola Pública": "Publica",
        "Escola JP II": "Publica",
        "Rede Decisão": "Privada_Bolsa",
        "Privada *Parcerias com Bolsa 100%": "Privada_Bolsa",
        "Privada - Programa de Apadrinhamento": "Privada_Bolsa",
        "Privada - Programa de apadrinhamento": "Privada_Bolsa",
        "Privada - Pagamento por *Empresa Parceira": "Privada_Bolsa",
        "Privada": "Privada",
        "Concluiu o 3º EM": "Concluido",
        "Bolsista Universitário *Formado (a)": "Concluido",
        "Nenhuma das opções acima": "Outro",
    }
    out = s.map(mapping_rules).fillna("Outro")
    out = out.where(s != "nan", other="Desconhecido")
    return out


def _coerce_pedra_when_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with NaN Pedra but a valid INDE: derive Pedra from PEDE thresholds.

    Pedra cutoffs documented by Passos Mágicos (PEDE 2024 dictionary):
      Quartzo < 5.948 ≤ Agata < 6.678 ≤ Ametista < 8.198 ≤ Topázio
    For rows where both Pedra and INDE are missing we leave Pedra as NaN — those
    rows are typically structurally inactive students that we still keep in the
    panel (e.g. for predictions in 2024) but cannot supervise.
    """
    mask = df["Pedra"].isna() & df["INDE"].notna()
    if mask.any():
        bins = [-np.inf, 5.948, 6.678, 8.198, np.inf]
        labels = ["Quartzo", "Agata", "Ametista", "Topázio"]
        derived = pd.cut(df.loc[mask, "INDE"], bins=bins, labels=labels)
        df.loc[mask, "Pedra"] = derived.astype(str)
        log.info("Derived Pedra from INDE thresholds for %d rows.", int(mask.sum()))
    return df


def load_raw(path: Optional[Path] = None) -> pd.DataFrame:
    """Read the CSV with conservative dtypes."""
    path = Path(path) if path else config.BASE_CSV
    log.info("Loading %s", path)
    df = pd.read_csv(path)
    log.info("Raw shape: %s", df.shape)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the canonical cleaning rules used by every downstream step."""
    df = df.copy()

    df["Pedra"] = _normalize_pedra(df["Pedra"])
    df["Gênero"] = _normalize_gender(df["Gênero"])
    df["Instituição de ensino"] = _normalize_instituicao(df["Instituição de ensino"])

    df = _coerce_pedra_when_missing(df)

    df["Pedra_ord"] = df["Pedra"].map(config.PEDRA_ORDER).astype("Float32")

    # Fase Ideal is currently a string like "Fase 7 (3º EM)" – extract numeric.
    df["FaseIdeal_num"] = (
        df["Fase Ideal"].astype(str).str.extract(r"Fase\s*(\d+)", expand=False)
        .astype("float").astype("Float32")
    )

    # Defasagem can be recomputed from Fase Ideal vs Fase for sanity but we
    # trust the column.
    df["Defasagem"] = df["Defasagem"].astype("Int16")
    df["Fase"] = df["Fase"].astype("Int16")
    df["Idade"] = df["Idade"].astype("Int16")
    df["Ano"] = df["Ano"].astype("Int16")

    # Drop any duplicate (RA, Ano) rows — the source promises uniqueness, but
    # we enforce it.
    before = len(df)
    df = df.drop_duplicates(subset=["RA", "Ano"], keep="first")
    if len(df) < before:
        log.warning("Dropped %d duplicate (RA, Ano) rows.", before - len(df))

    # Mark rows that have no useful indicator (INDE NaN AND no sub-indicator) –
    # we keep them in the panel so we can still produce predictions for them
    # via the latest year's metadata, but they will not enter training.
    indicator_present = df[INDICATOR_COLS].notna().any(axis=1)
    df["row_has_indicators"] = indicator_present

    log.info("Cleaned shape: %s (rows w/ indicators: %d)", df.shape, int(indicator_present.sum()))
    return df


def load_panel(path: Optional[Path] = None) -> pd.DataFrame:
    """Convenience: load + clean in one call."""
    return clean(load_raw(path))
