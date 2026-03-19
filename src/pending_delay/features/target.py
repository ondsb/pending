"""CLV target variable and delay tier classification."""

import logging

import pandas as pd

log = logging.getLogger(__name__)


def add_target(df: pd.DataFrame, target_col: str = "odds_after_10") -> pd.DataFrame:
    """Drop rows with null target."""
    mask = df[target_col].notna()
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        log.info(f"Dropping {n_dropped:,} rows with null {target_col} "
                 f"({n_dropped / len(df) * 100:.1f}%)")
    return df[mask].copy()


def classify_toxicity(
    predictions: pd.Series,
    pending: float = -0.02,
) -> pd.Series:
    """Map continuous CLV predictions to binary delay tiers: PENDING or SKIP.

    pred < pending  → PENDING (keep original delay)
    pred >= pending → SKIP (bypass delay)
    """
    tiers = pd.Series("SKIP", index=predictions.index)
    tiers[predictions < pending] = "PENDING"
    return tiers
