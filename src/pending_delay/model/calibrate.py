"""Calibration for the regression model.

For regression on continuous CLV, calibration means:
predicted CLV aligns with observed CLV across the prediction range.

We use isotonic regression to learn a monotonic mapping from raw predictions
to calibrated predictions, trained on the validation set.
"""

import json
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from pending_delay.config import settings
from pending_delay.features.engineering import prepare_features
from pending_delay.features.target import add_target
from pending_delay.model.train import temporal_split
from pending_delay.schema import TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def fit_calibrator(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> IsotonicRegression:
    """Fit isotonic regression calibrator on validation set."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_pred, y_true)
    return iso


def calibration_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Binned calibration report: group by predicted decile, compare means.

    Returns a DataFrame with columns:
        bin, n, mean_predicted, mean_actual, abs_error
    """
    df = pd.DataFrame({"pred": y_pred, "actual": y_true})
    df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")
    report = (
        df.groupby("bin", observed=True)
        .agg(
            n=("actual", "count"),
            mean_predicted=("pred", "mean"),
            mean_actual=("actual", "mean"),
        )
        .reset_index()
    )
    report["abs_error"] = np.abs(report["mean_predicted"] - report["mean_actual"])
    return report


def calibrate_model(
    data_path: Path,
    model_dir: Path | None = None,
) -> IsotonicRegression:
    """Load trained model, fit calibrator on validation set, save.

    Args:
        data_path: Path to merged parquet data.
        model_dir: Directory with model artifacts. Defaults to settings.model_dir.

    Returns:
        Fitted IsotonicRegression calibrator.
    """
    model_dir = model_dir or settings.model_dir

    # Load model
    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
    with open(model_dir / "cat_maps.json") as f:
        cat_maps = json.load(f)

    # Load and split data to get validation set
    log.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    df = add_target(df, TARGET_COL)
    _, val_df, _ = temporal_split(
        df,
        train_frac=settings.split.train_frac,
        val_frac=settings.split.val_frac,
    )

    # Prepare features and predict
    X_val, y_val, _ = prepare_features(val_df, fit_categories=False, category_maps=cat_maps)
    raw_preds = booster.predict(X_val)

    # Pre-calibration report
    log.info("Pre-calibration report:")
    pre_report = calibration_report(y_val.values, raw_preds)
    log.info(f"\n{pre_report.to_string(index=False)}")

    # Fit calibrator
    iso = fit_calibrator(y_val.values, raw_preds)
    cal_preds = iso.predict(raw_preds)

    # Post-calibration report
    log.info("Post-calibration report:")
    post_report = calibration_report(y_val.values, cal_preds)
    log.info(f"\n{post_report.to_string(index=False)}")

    # Metrics
    pre_mae = np.mean(np.abs(raw_preds - y_val.values))
    post_mae = np.mean(np.abs(cal_preds - y_val.values))
    log.info(f"MAE: pre={pre_mae:.5f} → post={post_mae:.5f}")

    # Save
    with open(model_dir / "calibrator.pkl", "wb") as f:
        pickle.dump(iso, f)
    pre_report.to_csv(model_dir / "calibration_pre.csv", index=False)
    post_report.to_csv(model_dir / "calibration_post.csv", index=False)
    log.info(f"Saved calibrator to {model_dir / 'calibrator.pkl'}")

    return iso


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate pending delay model")
    parser.add_argument(
        "--data",
        type=Path,
        default=settings.data_dir / "tickets.parquet",
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()

    calibrate_model(args.data, args.model_dir)


if __name__ == "__main__":
    main()
