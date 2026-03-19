"""Isotonic regression calibration on the saved validation set."""

import json
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from pending_delay.config import settings
from pending_delay.features.engineering import encode_features_to_numpy
from pending_delay.schema import TARGET_COL

log = logging.getLogger(__name__)


def fit_calibrator(y_true: np.ndarray, y_pred: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression calibrator."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_pred, y_true)
    return iso


def calibration_report(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Binned calibration report: group by predicted decile, compare means."""
    df = pd.DataFrame({"pred": y_pred, "actual": y_true})
    df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")
    report = (
        df.groupby("bin", observed=True)
        .agg(n=("actual", "count"), mean_predicted=("pred", "mean"), mean_actual=("actual", "mean"))
        .reset_index()
    )
    report["abs_error"] = np.abs(report["mean_predicted"] - report["mean_actual"])
    return report


def calibrate_model(model_dir: Path | None = None) -> IsotonicRegression:
    """Load trained model, fit calibrator on saved validation set.

    Args:
        model_dir: Directory with model artifacts (model.txt, val_set.parquet).

    Returns:
        Fitted IsotonicRegression calibrator.
    """
    model_dir = model_dir or settings.model_dir

    # Load model
    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))

    # Load saved validation set (written by train.py)
    val_path = model_dir / "val_set.parquet"
    if not val_path.exists():
        raise FileNotFoundError(
            f"No validation set at {val_path}. Re-run training first — "
            "the updated train.py saves val_set.parquet automatically."
        )
    # Val parquet already has engineered features from split_to_parquet
    val_df = pd.read_parquet(val_path)
    log.info(f"Loaded validation set: {len(val_df):,} rows")

    y_val = val_df[TARGET_COL].values
    with open(model_dir / "feature_names.json") as f:
        expected_features = json.load(f)
    X_val = encode_features_to_numpy(val_df[expected_features])
    raw_preds = booster.predict(X_val)

    # Pre-calibration report
    log.info("Pre-calibration report:")
    pre_report = calibration_report(y_val, raw_preds)
    log.info(f"\n{pre_report.to_string(index=False)}")

    # Fit calibrator
    iso = fit_calibrator(y_val, raw_preds)
    cal_preds = iso.predict(raw_preds)

    # Post-calibration report
    log.info("Post-calibration report:")
    post_report = calibration_report(y_val, cal_preds)
    log.info(f"\n{post_report.to_string(index=False)}")

    # Metrics
    pre_mae = float(np.mean(np.abs(raw_preds - y_val)))
    post_mae = float(np.mean(np.abs(cal_preds - y_val)))
    log.info(f"MAE: pre={pre_mae:.5f} -> post={post_mae:.5f}")

    # Save
    with open(model_dir / "calibrator.pkl", "wb") as f:
        pickle.dump(iso, f)
    pre_report.to_csv(model_dir / "calibration_pre.csv", index=False)
    post_report.to_csv(model_dir / "calibration_post.csv", index=False)
    log.info(f"Saved calibrator to {model_dir / 'calibrator.pkl'}")

    return iso


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Calibrate pending delay model")
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()
    calibrate_model(args.model_dir)


if __name__ == "__main__":
    main()
