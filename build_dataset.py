"""Pipeline runner: inspect data, train, calibrate, and run OPE."""

import argparse
import logging
from pathlib import Path

import duckdb
import pandas as pd

from pending_delay.config import settings
from pending_delay.schema import TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def inspect_data(data_path: Path) -> None:
    """Print summary statistics about the dataset, focusing on target distribution.

    Uses DuckDB to compute stats without loading the full dataset into memory.
    """
    con = duckdb.connect()
    con.sql("SET memory_limit = '2GB'")

    shape = con.sql(f"""
        SELECT COUNT(*) AS rows, COUNT(COLUMNS(*)) AS cols
        FROM read_parquet('{data_path}')
    """).fetchone()
    cols = [
        r[0]
        for r in con.sql(f"""
        SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{data_path}'))
    """).fetchall()
    ]
    log.info(f"Shape: ({shape[0]:,}, {len(cols)})")
    log.info(f"Columns: {cols}")

    n_total = shape[0]
    if TARGET_COL in cols:
        stats = con.sql(f"""
            SELECT
                COUNT(*) AS n_valid,
                AVG({TARGET_COL}) AS mean_val,
                MEDIAN({TARGET_COL}) AS median_val,
                STDDEV({TARGET_COL}) AS std_val,
                MIN({TARGET_COL}) AS min_val,
                MAX({TARGET_COL}) AS max_val
            FROM read_parquet('{data_path}')
            WHERE {TARGET_COL} IS NOT NULL AND isfinite({TARGET_COL})
        """).fetchone()
        n_valid, mean_v, median_v, std_v, min_v, max_v = stats
        log.info(f"\n=== Target: {TARGET_COL} ===")
        log.info(
            f"  Non-null & finite: {n_valid:,} / {n_total:,} ({n_valid / n_total * 100:.1f}%)"
        )
        log.info(f"  Mean:   {mean_v:.5f}")
        log.info(f"  Median: {median_v:.5f}")
        log.info(f"  Std:    {std_v:.5f}")
        log.info(f"  Min:    {min_v:.5f}")
        log.info(f"  Max:    {max_v:.5f}")

        # Distribution of toxicity
        thresholds = settings.thresholds
        tier_stats = con.sql(f"""
            SELECT
                SUM(CASE WHEN {TARGET_COL} < {thresholds.higher} THEN 1 ELSE 0 END) AS n_higher,
                SUM(CASE WHEN {TARGET_COL} >= {thresholds.higher} AND {TARGET_COL} < {thresholds.static_lower} THEN 1 ELSE 0 END) AS n_static,
                SUM(CASE WHEN {TARGET_COL} >= {thresholds.static_lower} AND {TARGET_COL} < {thresholds.lower_skip} THEN 1 ELSE 0 END) AS n_lower,
                SUM(CASE WHEN {TARGET_COL} >= {thresholds.lower_skip} THEN 1 ELSE 0 END) AS n_skip
            FROM read_parquet('{data_path}')
            WHERE {TARGET_COL} IS NOT NULL
        """).fetchone()
        n_higher, n_static, n_lower, n_skip = tier_stats

        log.info(f"\n=== Tier distribution (with current thresholds) ===")
        log.info(
            f"  HIGHER (< {thresholds.higher}):   {n_higher:>8,} ({n_higher / n_valid * 100:5.1f}%)"
        )
        log.info(
            f"  STATIC:                         {n_static:>8,} ({n_static / n_valid * 100:5.1f}%)"
        )
        log.info(
            f"  LOWER:                          {n_lower:>8,} ({n_lower / n_valid * 100:5.1f}%)"
        )
        log.info(
            f"  SKIP   (>= {thresholds.lower_skip}):  {n_skip:>8,} ({n_skip / n_valid * 100:5.1f}%)"
        )

    # Check for ticket_state distribution
    if "ticket_state" in cols:
        log.info(f"\n=== ticket_state ===")
        ts = con.sql(f"""
            SELECT ticket_state, COUNT(*) AS cnt
            FROM read_parquet('{data_path}')
            GROUP BY ticket_state ORDER BY cnt DESC
        """).fetchall()
        for state, cnt in ts:
            log.info(f"  {state}: {cnt:,}")

    # Pre-training filter impact
    filter_rules = settings.filter.rules
    if filter_rules:
        log.info(f"\n=== Pre-training filters ({len(filter_rules)} rules) ===")
        for rule in filter_rules:
            if rule.column not in cols:
                log.info(
                    f"  {rule.column} {rule.op} {rule.value}  — column NOT in data, will be skipped"
                )
                continue
            # Build a WHERE clause that selects rows that would be DROPPED
            # (i.e. rows that do NOT satisfy the keep condition).
            negate = {
                ">=": "<",
                ">": "<=",
                "<=": ">",
                "<": ">=",
                "==": "!=",
                "!=": "==",
            }
            neg_op = negate.get(rule.op, rule.op)
            val = f"'{rule.value}'" if isinstance(rule.value, str) else rule.value
            drop_result = con.sql(f"""
                SELECT COUNT(*) FROM read_parquet('{data_path}')
                WHERE {rule.column} {neg_op} {val} OR {rule.column} IS NULL
            """).fetchone()
            n_drop = drop_result[0]
            log.info(
                f"  {rule.column} {rule.op} {rule.value}  — would drop {n_drop:,} / {shape[0]:,} "
                f"rows ({n_drop / shape[0] * 100:.1f}%)"
            )

    # Null counts for aggregate features
    agg_cols = [
        c
        for c in cols
        if c.startswith("bs_")
        or c.startswith("total_")
        or c.startswith("avg_rejected")
        or c.startswith("risk_tier")
        or c == "mean_stake_size"
    ]
    if agg_cols:
        null_exprs = ", ".join(
            f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS {c}" for c in agg_cols
        )
        nulls = con.sql(
            f"SELECT {null_exprs} FROM read_parquet('{data_path}')"
        ).fetchone()
        log.info(f"\n=== Aggregate feature nulls ===")
        for col, n in zip(agg_cols, nulls):
            log.info(f"  {col}: {n:,}")

    con.close()


def main():
    parser = argparse.ArgumentParser(description="Build and train pending delay model")
    parser.add_argument(
        "--data",
        type=Path,
        default=settings.data_dir / "tickets.parquet",
        help="Path to merged parquet data",
    )
    parser.add_argument(
        "--inspect", action="store_true", help="Just inspect data, don't train"
    )
    parser.add_argument(
        "--full", action="store_true", help="Run full pipeline: train + calibrate + OPE"
    )
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

        calibrate_model(args.model_dir)

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
