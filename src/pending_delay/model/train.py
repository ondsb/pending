"""LightGBM training with temporal split and calibration evaluation.

Usage:
    python -m pending_delay.model.train --data data/tickets.parquet
"""

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from pending_delay.config import settings
from pending_delay.features.engineering import (
    get_categorical_indices,
    prepare_features,
)
from pending_delay.features.target import add_target
from pending_delay.schema import TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def temporal_split(
    df: pd.DataFrame,
    time_col: str = "created_at",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe by time: oldest train, middle val, newest test."""
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    log.info(
        f"Temporal split: train={len(train):,} [{df[time_col].iloc[0]} → {df[time_col].iloc[train_end-1]}], "
        f"val={len(val):,}, test={len(test):,}"
    )
    return train, val, test


def train_model(
    data_path: Path,
    model_dir: Path | None = None,
) -> tuple[lgb.Booster, dict]:
    """Train LightGBM regression model on merged parquet data.

    Args:
        data_path: Path to merged parquet file.
        model_dir: Directory to save model artifacts. Defaults to settings.model_dir.

    Returns:
        Tuple of (trained booster, metrics dict).
    """
    model_dir = model_dir or settings.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    log.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    log.info(f"  {len(df):,} rows x {len(df.columns)} cols")

    # Filter to rows with valid target
    df = add_target(df, TARGET_COL)
    log.info(f"  {len(df):,} rows after dropping null targets")

    # Temporal split
    train_df, val_df, test_df = temporal_split(
        df,
        train_frac=settings.split.train_frac,
        val_frac=settings.split.val_frac,
    )

    # Save test set for OPE (includes metadata columns we'll need)
    test_path = model_dir / "test_set.parquet"
    test_df.to_parquet(test_path, index=False)
    log.info(f"  Saved test set to {test_path}")

    # Prepare features
    X_train, y_train, cat_maps = prepare_features(train_df, fit_categories=True)
    X_val, y_val, _ = prepare_features(val_df, fit_categories=False, category_maps=cat_maps)

    feature_names = X_train.columns.tolist()
    cat_indices = get_categorical_indices(feature_names)
    log.info(f"  Features: {len(feature_names)}, Categoricals: {len(cat_indices)}")
    log.info(f"  Target stats — train: mean={y_train.mean():.4f} std={y_train.std():.4f}")
    log.info(f"  Target stats — val:   mean={y_val.mean():.4f} std={y_val.std():.4f}")

    # Build LightGBM datasets
    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=cat_indices,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_val,
        label=y_val,
        categorical_feature=cat_indices,
        reference=dtrain,
        free_raw_data=False,
    )

    # Train
    params = {
        "objective": settings.model.objective,
        "metric": settings.model.metric,
        "num_leaves": settings.model.num_leaves,
        "learning_rate": settings.model.learning_rate,
        "feature_fraction": settings.model.feature_fraction,
        "bagging_fraction": settings.model.bagging_fraction,
        "bagging_freq": settings.model.bagging_freq,
        "min_child_samples": settings.model.min_child_samples,
        "verbose": settings.model.verbose,
    }

    log.info("Training LightGBM...")
    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=settings.model.early_stopping_rounds),
    ]

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=settings.model.n_estimators,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # Evaluate
    val_preds = booster.predict(X_val)
    mae = np.mean(np.abs(val_preds - y_val))
    rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))

    # Feature importance
    importance = dict(
        zip(feature_names, booster.feature_importance(importance_type="gain").tolist())
    )
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    metrics = {
        "val_mae": float(mae),
        "val_rmse": float(rmse),
        "best_iteration": booster.best_iteration,
        "n_features": len(feature_names),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(test_df),
        "feature_importance_top20": dict(list(importance.items())[:20]),
    }

    log.info(f"  Val MAE: {mae:.5f}, Val RMSE: {rmse:.5f}")
    log.info(f"  Best iteration: {booster.best_iteration}")
    log.info(f"  Top 5 features: {list(importance.keys())[:5]}")

    # Save artifacts
    booster.save_model(str(model_dir / "model.txt"))
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(model_dir / "cat_maps.json", "w") as f:
        json.dump(cat_maps, f)
    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    log.info(f"  Saved model artifacts to {model_dir}")

    return booster, metrics


def main():
    parser = argparse.ArgumentParser(description="Train pending delay model")
    parser.add_argument(
        "--data",
        type=Path,
        default=settings.data_dir / "tickets.parquet",
        help="Path to merged parquet data",
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()

    train_model(args.data, args.model_dir)


if __name__ == "__main__":
    main()
