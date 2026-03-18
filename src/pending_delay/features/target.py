"""CLV-based target variable definition.

The target is `odds_after_10`: the line movement 10 seconds after ticket placement.
- Negative values = line moved in bettor's favor = sharp/toxic bet
- Positive values = line moved against bettor = recreational bet
- Near zero = no significant movement

This is used as a continuous regression target. At inference time, predicted CLV
is mapped to delay tiers via tunable thresholds (no retraining needed).
"""

import pandas as pd


def add_target(df: pd.DataFrame, target_col: str = "odds_after_10") -> pd.DataFrame:
    """Add the regression target column from raw odds_after_10.

    The target IS odds_after_10 directly — we use it as a continuous regression target.
    Rows with null target are dropped (can't train on them).
    """
    mask = df[target_col].notna()
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        print(f"[target] Dropping {n_dropped:,} rows with null {target_col} "
              f"({n_dropped / len(df) * 100:.1f}%)")
    return df[mask].copy()


def classify_toxicity(
    predictions: pd.Series,
    higher: float = -0.02,
    static_lower: float = -0.005,
    lower_skip: float = 0.005,
) -> pd.Series:
    """Map continuous CLV predictions to delay tiers.

    Args:
        predictions: Predicted odds_after_10 values.
        higher: Threshold below which → HIGHER (punitive delay).
        static_lower: Threshold below which → STATIC (unchanged).
        lower_skip: Threshold below which → LOWER (reduced delay).
            Above this → SKIP (bypass).

    Returns:
        Series of tier labels: HIGHER, STATIC, LOWER, SKIP.
    """
    tiers = pd.Series("SKIP", index=predictions.index)
    tiers[predictions < lower_skip] = "LOWER"
    tiers[predictions < static_lower] = "STATIC"
    tiers[predictions < higher] = "HIGHER"
    return tiers
