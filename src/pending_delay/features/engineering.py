"""Feature engineering and encoding utilities."""

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
