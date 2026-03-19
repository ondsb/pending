"""Feature engineering and encoding utilities."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the dataframe."""
    df = df.copy()

    if "mean_stake_size" in df.columns and "stake" in df.columns:
        df["stake_ratio"] = np.where(
            df["mean_stake_size"].notna() & (df["mean_stake_size"] > 0),
            df["stake"] / df["mean_stake_size"],
            np.nan,
        )

    if "selection_odds" in df.columns:
        df["odds_bucket"] = pd.cut(
            df["selection_odds"],
            bins=[0, 1.3, 1.7, 2.2, 3.5, float("inf")],
            labels=["heavy_fav", "slight_fav", "even", "underdog", "longshot"],
            ordered=False,
        ).astype("object")

    return df


def encode_features_to_numpy(df: pd.DataFrame) -> np.ndarray:
    """Encode a DataFrame of features to a float64 numpy array.

    Object/string columns are converted to integer category codes.
    Used by training, calibration, and OPE to ensure consistent encoding.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("__missing__").astype("category").cat.codes
    return df.to_numpy(dtype=np.float64)


def prepare_features(
    df: pd.DataFrame,
    fit_categories: bool = True,
    category_maps: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Prepare a DataFrame for model training or inference.

    Parameters
    ----------
    df:
        DataFrame containing feature columns **and** the target column.
        Null targets should already be filtered (see ``add_target``).
    fit_categories:
        If *True*, learn category mappings from the data (training mode).
        If *False*, apply the supplied *category_maps* (inference mode).
    category_maps:
        Pre-fitted category mappings ``{col: [cat1, cat2, ...]}``.
        Required when ``fit_categories=False``.

    Returns
    -------
    (X, y, cat_maps):
        X  – DataFrame of feature columns only (categoricals as object dtype)
        y  – Series of target values
        cat_maps – dict mapping each categorical column to its known values
    """
    from pending_delay.config import settings
    from pending_delay.schema import (
        CATEGORICAL_FEATURES,
        DROP_METADATA,
        LEAKED_COLUMNS,
        TARGET_COL,
    )

    df = engineer_features(df)

    y = df[TARGET_COL].copy()

    # Select feature columns
    configured = settings.feature.features
    if configured:
        features = [f for f in configured if f in df.columns]
    else:
        exclude = set(LEAKED_COLUMNS + DROP_METADATA + [TARGET_COL])
        features = [c for c in df.columns if c not in exclude]

    X = df[features].copy()

    # Handle categoricals
    cats = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    new_maps: dict[str, list[str]] = {}

    if fit_categories:
        for col in cats:
            known = sorted(X[col].dropna().unique().tolist())
            new_maps[col] = known
            X[col] = X[col].fillna("__missing__")
    else:
        maps = category_maps or {}
        for col in cats:
            X[col] = X[col].fillna("__missing__")
            if col in maps:
                known = set(maps[col])
                X[col] = X[col].where(X[col].isin(known), "__missing__")
        new_maps = maps

    return X, y, new_maps


def get_categorical_indices(feature_names: list[str]) -> list[int]:
    """Return the positional indices of categorical features in *feature_names*."""
    from pending_delay.schema import CATEGORICAL_FEATURES

    return [i for i, f in enumerate(feature_names) if f in CATEGORICAL_FEATURES]
