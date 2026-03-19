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
    higher: float = -0.02,
    static_lower: float = -0.005,
    lower_skip: float = 0.005,
) -> pd.Series:
    """Map continuous CLV predictions to delay tiers: HIGHER, STATIC, LOWER, SKIP."""
    tiers = pd.Series("SKIP", index=predictions.index)
    tiers[predictions < lower_skip] = "LOWER"
    tiers[predictions < static_lower] = "STATIC"
    tiers[predictions < higher] = "HIGHER"
    return tiers
