"""Validate an external LightGBM model against a parquet dataset.

This is the primary entry point for evaluating pre-trained models that were
NOT trained by this project's pipeline.  It sets up a model directory with
all the artifacts the Streamlit dashboard expects, then runs calibration and
Offline Policy Evaluation (OPE).

Usage examples
--------------
# Validate a single model (auto-name from filename):
python validate_model.py --model /path/to/model.txt --data data/tickets.parquet

# Give it a name and custom val/test split:
python validate_model.py \
    --model /path/to/model.txt \
    --data data/tickets.parquet \
    --name clv_v3 \
    --val-frac 0.2

# Skip calibration (use raw predictions):
python validate_model.py \
    --model /path/to/model.txt \
    --data data/tickets.parquet \
    --no-calibrate

# Supply known feature names:
python validate_model.py \
    --model /path/to/model.txt \
    --data data/tickets.parquet \
    --feature-names feature_names.json
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pending_delay.config import settings
from pending_delay.features.engineering import engineer_features
from pending_delay.features.target import add_target
from pending_delay.schema import TARGET_COL

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_feature_names(
    model_path: Path,
    feature_names_path: Path | None = None,
) -> list[str]:
    """Extract feature names from a JSON file or the LightGBM model file.

    Raises ``SystemExit`` if the model has no embedded feature names and
    no ``--feature-names`` JSON was supplied.  Silently guessing the wrong
    feature mapping produces garbage predictions, so we fail early instead.
    """
    # 1. Explicit JSON takes priority
    if feature_names_path is not None:
        with open(feature_names_path) as f:
            names = json.load(f)
        log.info(f"Loaded {len(names)} feature names from {feature_names_path}")
        return names

    # 2. Try to read from the model itself
    booster = lgb.Booster(model_file=str(model_path))
    names = booster.feature_name()
    n_features = booster.num_feature()

    # LightGBM uses "Column_N" as placeholder when names weren't saved
    if names and not names[0].startswith("Column_"):
        log.info(f"Inferred {len(names)} feature names from model file")
        return names

    # 3. Model has no real feature names — refuse to guess
    log.error(
        f"Model has {n_features} features but no feature names "
        f"(stored as Column_0 … Column_{n_features - 1}).\n"
        f"  Cannot determine the correct column mapping — using the wrong\n"
        f"  features (or the wrong order) will produce silent garbage.\n\n"
        f"  Please supply a feature_names.json via --feature-names:\n"
        f"    python validate_model.py --model {model_path} "
        f"--feature-names feature_names.json ...\n"
    )
    sys.exit(1)


def _apply_pretraining_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same pre-training row filters used during model training."""
    n_before = len(df)
    for rule in settings.filter.rules:
        col = rule.column
        if col not in df.columns:
            continue
        val = rule.value
        ops = {
            ">=": lambda s, v: s >= v,
            ">": lambda s, v: s > v,
            "<=": lambda s, v: s <= v,
            "<": lambda s, v: s < v,
            "==": lambda s, v: s == v,
            "!=": lambda s, v: s != v,
        }
        op_fn = ops.get(rule.op)
        if op_fn is not None:
            mask = op_fn(df[col], val)
            df = df[mask]
    n_after = len(df)
    if n_before != n_after:
        log.info(
            f"Pre-training filters: {n_before:,} -> {n_after:,} rows "
            f"(dropped {n_before - n_after:,})"
        )
    return df


def _temporal_split(
    df: pd.DataFrame,
    val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into validation and test sets using a temporal ordering.

    The data is assumed to already be ordered by time (``created_at``).
    Returns (val_df, test_df).
    """
    n = len(df)
    val_end = int(n * val_frac)
    return df.iloc[:val_end].copy(), df.iloc[val_end:].copy()


# ---------------------------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------------------------


def validate_model(
    model_path: Path,
    data_path: Path,
    name: str | None = None,
    feature_names_path: Path | None = None,
    val_frac: float = 0.3,
    calibrate: bool = True,
    max_row_groups: int | None = None,
) -> Path:
    """Validate an external LightGBM model against *data_path*.

    Creates a fully-populated model directory under ``models/<name>/``
    that the Streamlit dashboard can discover and display.

    Parameters
    ----------
    model_path:
        Path to a LightGBM text-format model (``model.txt``).
    data_path:
        Path to the tickets parquet file.
    name:
        Human-readable model name (used as directory name).  Defaults to
        the model file's parent directory name.
    feature_names_path:
        Optional JSON file listing feature column names.  If omitted,
        names are inferred from the model file or the project config.
    val_frac:
        Fraction of data to hold out for calibration (default 0.3).
        The remaining ``1 - val_frac`` is used for evaluation.
    calibrate:
        Whether to fit an isotonic calibrator on the validation set.
    max_row_groups:
        If set, read only the last *N* row groups from the parquet file
        instead of loading everything.  Useful for machines with limited
        RAM (e.g. the full dataset may exceed available memory).

    Returns
    -------
    Path to the created model directory.
    """
    # --- Resolve name and output directory ---
    if name is None:
        name = model_path.parent.name
        if name in (".", ""):
            name = model_path.stem
    model_dir = settings.model_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Model directory: {model_dir}")

    # --- Copy model file ---
    dest_model = model_dir / "model.txt"
    if model_path.resolve() != dest_model.resolve():
        shutil.copy2(model_path, dest_model)
        log.info(f"Copied model to {dest_model}")

    # --- Determine feature names ---
    feature_names = _infer_feature_names(model_path, feature_names_path)

    fn_path = model_dir / "feature_names.json"
    with open(fn_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    # --- Load and prepare data ---
    log.info(f"Loading data from {data_path}")
    pf = pq.ParquetFile(data_path)
    total_rows = pf.metadata.num_rows
    n_row_groups = pf.metadata.num_row_groups
    log.info(
        f"  {total_rows:,} total rows, {pf.metadata.num_columns} columns, "
        f"{n_row_groups} row groups"
    )

    if max_row_groups is not None and max_row_groups < n_row_groups:
        # Read only the last N row groups to stay within RAM limits.
        # The data is stored in temporal order, so the last groups are the
        # most recent — exactly what we want for validation/test splits.
        start_rg = n_row_groups - max_row_groups
        tables = [pf.read_row_group(i) for i in range(start_rg, n_row_groups)]
        import pyarrow as pa

        df = pa.concat_tables(tables).to_pandas()
        del tables  # free Arrow memory
        log.info(
            f"  Loaded last {max_row_groups} row groups: {len(df):,} rows "
            f"(skipped first {start_rg} groups)"
        )
    else:
        df = pd.read_parquet(data_path)

    # Engineer features (adds stake_ratio, odds_bucket)
    df = engineer_features(df)

    # Apply pre-training filters
    df = _apply_pretraining_filters(df)

    # Sort by time if possible (ensures temporal split is meaningful)
    if "created_at" in df.columns:
        df = df.sort_values("created_at").reset_index(drop=True)

    # Drop rows with null target
    df = add_target(df, TARGET_COL)
    log.info(f"  {len(df):,} rows after filtering + null target removal")

    # Verify feature columns exist
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        log.warning(
            f"Features missing from data ({len(missing)}): {missing}. "
            f"Predictions may be incorrect."
        )
    available_features = [f for f in feature_names if f in df.columns]
    if not available_features:
        log.error("No feature columns found in data.  Cannot proceed.")
        sys.exit(1)

    # --- Split ---
    val_df, test_df = _temporal_split(df, val_frac)
    log.info(f"  Val: {len(val_df):,}  Test: {len(test_df):,}")

    val_df.to_parquet(model_dir / "val_set.parquet", index=False)
    test_df.to_parquet(model_dir / "test_set.parquet", index=False)

    # --- Build category maps from val set (for dashboard loader) ---
    from pending_delay.schema import CATEGORICAL_FEATURES

    cat_maps: dict[str, list[str]] = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            cat_maps[col] = sorted(df[col].dropna().unique().tolist())
    with open(model_dir / "cat_maps.json", "w") as f:
        json.dump(cat_maps, f, indent=2)

    # --- Calibration (optional) ---
    if calibrate:
        log.info("Running calibration on validation set...")
        from pending_delay.model.calibrate import calibrate_model

        calibrate_model(model_dir)
        log.info("  Calibration complete")
    else:
        log.info("Skipping calibration (--no-calibrate)")

    # --- OPE on test set ---
    log.info("Running Offline Policy Evaluation on test set...")
    from pending_delay.evaluation.ope import run_ope

    ope_metrics = run_ope(model_dir)

    # --- Generate metrics.json (the dashboard Model Results tab reads this) ---
    booster = lgb.Booster(model_file=str(dest_model))

    # Compute validation-set metrics for the dashboard
    from pending_delay.features.engineering import encode_features_to_numpy
    from pending_delay.evaluation.metrics import regression_metrics

    X_val = encode_features_to_numpy(val_df[available_features])
    raw_val_preds = booster.predict(X_val)

    # Apply calibrator if it exists
    import pickle

    cal_path = model_dir / "calibrator.pkl"
    if cal_path.exists():
        with open(cal_path, "rb") as f:
            calibrator = pickle.load(f)
        val_preds = calibrator.predict(raw_val_preds)
    else:
        val_preds = raw_val_preds

    y_val = val_df[TARGET_COL].values
    val_metrics = regression_metrics(y_val, val_preds)

    # Feature importance
    importance_raw = dict(
        zip(
            feature_names,
            booster.feature_importance(importance_type="gain").tolist(),
        )
    )
    importance_sorted = dict(sorted(importance_raw.items(), key=lambda x: -x[1]))

    metrics = {
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "best_iteration": booster.best_iteration,
        "n_features": len(feature_names),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "features_used": feature_names,
        "feature_importance_top20": dict(list(importance_sorted.items())[:20]),
        "source": "external",
    }
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Summary ---
    print()
    print("=" * 60)
    print(f"  Validation complete: {name}")
    print("=" * 60)
    print(f"  Model dir:   {model_dir}")
    print(f"  Features:    {len(feature_names)}")
    print(f"  Val rows:    {len(val_df):,}")
    print(f"  Test rows:   {len(test_df):,}")
    print(f"  Calibrated:  {'yes' if calibrate else 'no'}")
    print()
    print("  Validation set:")
    print(f"    MAE:   {val_metrics['mae']:.5f}")
    print(f"    RMSE:  {val_metrics['rmse']:.5f}")
    print()
    print("  Test set (OPE):")
    print(f"    MAE:   {ope_metrics['mae']:.5f}")
    print(f"    RMSE:  {ope_metrics['rmse']:.5f}")
    print(f"    Corr:  {ope_metrics['correlation']:.4f}")
    print()
    print("  Policy simulation:")
    print(f"    Skip rate:         {ope_metrics['skip_rate']:.1%}")
    print(f"    Friction reduced:  {ope_metrics['friction_reduced_rate']:.1%}")
    print(f"    HIGHER count:      {ope_metrics['higher_count']:,}")
    print(
        f"    PnL delta:         {ope_metrics['pnl_delta']:+,.0f} ({ope_metrics['pnl_delta_pct']:+.1f}%)"
    )
    print()
    print("  Run the dashboard:  streamlit run app.py")
    print()

    return model_dir


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Validate an external LightGBM model against ticket data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to a LightGBM text-format model file (model.txt)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=settings.data_dir / "tickets.parquet",
        help="Path to the tickets parquet file (default: data/tickets.parquet)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Model name (used as directory name under models/).  "
        "Defaults to the model file's parent directory name.",
    )
    parser.add_argument(
        "--feature-names",
        type=Path,
        default=None,
        dest="feature_names",
        help="JSON file listing feature column names.  "
        "If omitted, names are inferred from the model or project config.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.3,
        dest="val_frac",
        help="Fraction of data for calibration validation set (default: 0.3)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        default=False,
        dest="no_calibrate",
        help="Skip isotonic calibration (use raw model predictions)",
    )
    parser.add_argument(
        "--max-row-groups",
        type=int,
        default=None,
        dest="max_row_groups",
        help="Read only the last N row groups from the parquet file to limit "
        "memory usage.  Each group is ~123k rows for the default dataset.  "
        "E.g. --max-row-groups 5 loads ~615k rows instead of the full 150M.",
    )

    args = parser.parse_args()

    if not args.model.exists():
        parser.error(f"Model file not found: {args.model}")
    if not args.data.exists():
        parser.error(f"Data file not found: {args.data}")

    validate_model(
        model_path=args.model,
        data_path=args.data,
        name=args.name,
        feature_names_path=args.feature_names,
        val_frac=args.val_frac,
        calibrate=not args.no_calibrate,
        max_row_groups=args.max_row_groups,
    )


if __name__ == "__main__":
    main()
