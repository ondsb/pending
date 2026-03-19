"""Model discovery, loading, and prediction caching for the Streamlit UI.

All tabs share the same cached model artifacts and predictions so that
loading + inference happens exactly once per model directory selection.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.isotonic import IsotonicRegression

from pending_delay.features.engineering import (
    encode_features_to_numpy,
    engineer_features,
)
from pending_delay.features.target import add_target
from pending_delay.schema import TARGET_COL


def discover_model_dirs(root: Path) -> list[Path]:
    """Find directories under *root* that contain a trained model.

    A valid model directory must have ``model.txt``.  Results are sorted
    alphabetically (newest-first if directories are timestamp-named).
    """
    if not root.exists():
        return []
    dirs = sorted(
        [p.parent for p in root.rglob("model.txt")],
        key=lambda p: p.name,
        reverse=True,
    )
    # Deduplicate (rglob may find nested paths)
    seen: set[Path] = set()
    unique: list[Path] = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


@st.cache_resource
def load_model_artifacts(model_dir: str) -> dict:
    """Load all model artifacts from *model_dir*.

    Cached with ``@st.cache_resource`` so the booster and calibrator are
    loaded once and shared across reruns.

    Parameters
    ----------
    model_dir:
        String path (Streamlit cache keys must be hashable).

    Returns
    -------
    dict with keys:
        booster, calibrator (or None), cat_maps, feature_names, metrics
    """
    md = Path(model_dir)

    booster = lgb.Booster(model_file=str(md / "model.txt"))

    calibrator: IsotonicRegression | None = None
    cal_path = md / "calibrator.pkl"
    if cal_path.exists():
        with open(cal_path, "rb") as f:
            calibrator = pickle.load(f)

    cat_maps: dict = {}
    cm_path = md / "cat_maps.json"
    if cm_path.exists():
        with open(cm_path) as f:
            cat_maps = json.load(f)

    feature_names: list[str] = []
    fn_path = md / "feature_names.json"
    if fn_path.exists():
        with open(fn_path) as f:
            feature_names = json.load(f)

    metrics: dict = {}
    m_path = md / "metrics.json"
    if m_path.exists():
        with open(m_path) as f:
            metrics = json.load(f)

    return {
        "booster": booster,
        "calibrator": calibrator,
        "cat_maps": cat_maps,
        "feature_names": feature_names,
        "metrics": metrics,
    }


@st.cache_data
def compute_predictions(model_dir: str) -> pd.DataFrame:
    """Load the test set, run inference, and return a DataFrame with predictions.

    The returned DataFrame contains **all original test-set columns** plus:

    * ``raw_pred``  – uncalibrated LightGBM prediction
    * ``cal_pred``  – calibrated prediction (equals raw_pred when no calibrator)
    * ``actual``    – ground-truth CLV (``odds_after_10``)

    Cached with ``@st.cache_data`` so inference runs once per model directory.
    """
    md = Path(model_dir)
    arts = load_model_artifacts(model_dir)
    booster: lgb.Booster = arts["booster"]
    calibrator: IsotonicRegression | None = arts["calibrator"]
    cat_maps: dict = arts["cat_maps"]

    test_path = md / "test_set.parquet"
    if not test_path.exists():
        st.error(
            f"Test set not found at `{test_path}`. Run the training pipeline first."
        )
        st.stop()

    test_df = pd.read_parquet(test_path)

    # Engineer features (adds stake_ratio, odds_bucket if not already present)
    test_with_target = add_target(test_df, TARGET_COL)
    test_with_target = engineer_features(test_with_target)

    # Use the model's own feature names (not the config's feature list) and
    # encode to numpy -- mirrors the OPE flow in evaluation/ope.py and avoids
    # pandas categorical mismatches with externally-trained models.
    feature_names: list[str] = arts["feature_names"]
    available = [f for f in feature_names if f in test_with_target.columns]
    X_test = encode_features_to_numpy(test_with_target[available])

    y_test = test_with_target[TARGET_COL]

    raw_preds = booster.predict(X_test)
    cal_preds = calibrator.predict(raw_preds) if calibrator is not None else raw_preds

    # Attach predictions to the original rows (preserving all metadata columns)
    result = test_with_target.copy()
    result["raw_pred"] = raw_preds
    result["cal_pred"] = cal_preds
    result["actual"] = y_test.values

    return result


def load_ope_metrics(model_dir: str) -> dict | None:
    """Load pre-computed OPE metrics if available."""
    path = Path(model_dir) / "ope" / "ope_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None
