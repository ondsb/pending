"""
Build training dataset from merged parquet (local or S3).

This script is now a thin wrapper: the real feature engineering lives in
src/pending_delay/features/. This script handles loading the merged parquet
and running the full pipeline: train → calibrate → OPE.

Usage:
    # Train on local subset data:
    python build_dataset.py --data data/tickets.parquet

    # Full pipeline (train + calibrate + OPE):
    python build_dataset.py --data data/tickets.parquet --full

    # Just inspect the data:
    python build_dataset.py --data data/tickets.parquet --inspect
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from pending_delay.config import settings
from pending_delay.features.target import add_target
from pending_delay.schema import TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def inspect_data(data_path: Path) -> None:
    """Print summary statistics about the dataset, focusing on target distribution."""
    df = pd.read_parquet(data_path)
    log.info(f"Shape: {df.shape}")
    log.info(f"Columns: {list(df.columns)}")

    if TARGET_COL in df.columns:
        target = df[TARGET_COL].dropna()
        log.info(f"\n=== Target: {TARGET_COL} ===")
        log.info(f"  Non-null: {len(target):,} / {len(df):,} ({len(target)/len(df)*100:.1f}%)")
        log.info(f"  Mean:   {target.mean():.5f}")
        log.info(f"  Median: {target.median():.5f}")
        log.info(f"  Std:    {target.std():.5f}")
        log.info(f"  Min:    {target.min():.5f}")
        log.info(f"  Max:    {target.max():.5f}")

        # Distribution of toxicity
        thresholds = settings.thresholds
        n_higher = (target < thresholds.higher).sum()
        n_static = ((target >= thresholds.higher) & (target < thresholds.static_lower)).sum()
        n_lower = ((target >= thresholds.static_lower) & (target < thresholds.lower_skip)).sum()
        n_skip = (target >= thresholds.lower_skip).sum()

        log.info(f"\n=== Tier distribution (with current thresholds) ===")
        log.info(f"  HIGHER (< {thresholds.higher}):   {n_higher:>8,} ({n_higher/len(target)*100:5.1f}%)")
        log.info(f"  STATIC:                         {n_static:>8,} ({n_static/len(target)*100:5.1f}%)")
        log.info(f"  LOWER:                          {n_lower:>8,} ({n_lower/len(target)*100:5.1f}%)")
        log.info(f"  SKIP   (>= {thresholds.lower_skip}):  {n_skip:>8,} ({n_skip/len(target)*100:5.1f}%)")

    # Check for ticket_state distribution
    if "ticket_state" in df.columns:
        log.info(f"\n=== ticket_state ===")
        log.info(f"{df['ticket_state'].value_counts().to_string()}")

    # Null counts for aggregate features
    agg_cols = [c for c in df.columns if c.startswith("bs_") or c.startswith("total_") or
                c.startswith("avg_rejected") or c.startswith("risk_tier") or c == "mean_stake_size"]
    if agg_cols:
        nulls = df[agg_cols].isnull().sum()
        log.info(f"\n=== Aggregate feature nulls ===")
        log.info(f"{nulls.to_string()}")


def main():
    parser = argparse.ArgumentParser(description="Build and train pending delay model")
    parser.add_argument(
        "--data", type=Path, default=settings.data_dir / "tickets.parquet",
        help="Path to merged parquet data",
    )
    parser.add_argument("--inspect", action="store_true", help="Just inspect data, don't train")
    parser.add_argument("--full", action="store_true", help="Run full pipeline: train + calibrate + OPE")
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.inspect:
        inspect_data(args.data)
        return

    # Train
    from pending_delay.model.train import train_model
    booster, metrics = train_model(args.data, args.model_dir)

    if args.full:
        # Calibrate
        from pending_delay.model.calibrate import calibrate_model
        calibrate_model(args.data, args.model_dir)

        # OPE
        from pending_delay.evaluation.ope import run_ope
        ope_metrics = run_ope(args.model_dir)
        log.info(f"\n=== OPE Results ===")
        for k, v in ope_metrics.items():
            if isinstance(v, float):
                log.info(f"  {k}: {v:.4f}")
            else:
                log.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
