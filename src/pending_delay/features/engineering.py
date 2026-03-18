"""Feature engineering pipeline.

Handles:
1. Dropping leaked/metadata columns
2. Engineering new features (stake ratio, odds bucket)
3. Preparing categorical features for LightGBM
"""

import numpy as np
import pandas as pd

from pending_delay.schema import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    DROP_METADATA,
    LEAKED_COLUMNS,
    TARGET_COL,
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the dataframe.

    Args:
        df: Raw merged dataframe (tickets + aggregates).

    Returns:
        DataFrame with engineered features added.
    """
    df = df.copy()

    # Stake ratio: is this bet unusually large for this bettor?
    df["stake_ratio"] = np.where(
        df["mean_stake_size"].notna() & (df["mean_stake_size"] > 0),
        df["stake"] / df["mean_stake_size"],
        np.nan,
    )

    # Odds bucket: bin selection_odds into ranges
    df["odds_bucket"] = pd.cut(
        df["selection_odds"],
        bins=[0, 1.3, 1.7, 2.2, 3.5, float("inf")],
        labels=["heavy_fav", "slight_fav", "even", "underdog", "longshot"],
        ordered=False,
    ).astype("object")  # LightGBM wants object or str for categoricals, not pd.Categorical

    return df


def prepare_features(
    df: pd.DataFrame,
    fit_categories: bool = True,
    category_maps: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]]]:
    """Full feature preparation pipeline: engineer, drop leaked, select features.

    Args:
        df: Raw merged dataframe with target still present.
        fit_categories: If True, learn category mappings from data.
        category_maps: Pre-fitted category mappings (for val/test sets).

    Returns:
        Tuple of (features_df, target_series, category_maps).
    """
    # Extract target before dropping leaked columns
    target = df[TARGET_COL].copy()

    # Engineer new features
    df = engineer_features(df)

    # Select only the features we want (automatically excludes leaked + metadata)
    available = [f for f in ALL_FEATURES if f in df.columns]
    features = df[available].copy()

    # Handle categoricals — convert to LightGBM-compatible codes
    cat_maps = category_maps or {}
    for col in CATEGORICAL_FEATURES:
        if col not in features.columns:
            continue
        features[col] = features[col].fillna("__missing__").astype(str)
        if fit_categories:
            cat_maps[col] = sorted(features[col].unique().tolist())

    return features, target, cat_maps


def get_feature_names() -> list[str]:
    """Return ordered list of all feature names."""
    return list(ALL_FEATURES)


def get_categorical_indices(feature_names: list[str]) -> list[int]:
    """Return indices of categorical features within the feature list."""
    return [
        feature_names.index(c)
        for c in CATEGORICAL_FEATURES
        if c in feature_names
    ]
