"""Offline Policy Evaluation (OPE) — the deliverable.

Maps calibrated predictions → delay tiers → simulates PnL impact.
Produces the key metrics: skip rate, friction reduction, PnL delta,
and stratified HIGHER tickets for manual review.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from pending_delay.config import settings
from pending_delay.evaluation.metrics import (
    plot_calibration_bins,
    plot_feature_importance,
    plot_predicted_vs_actual,
    regression_metrics,
)
from pending_delay.evaluation.simulate import (
    assign_tiers,
    policy_summary,
    simulate_policy,
)
from pending_delay.features.engineering import prepare_features
from pending_delay.features.target import add_target
from pending_delay.schema import TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_ope(
    model_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Run full OPE pipeline on the held-out test set.

    Args:
        model_dir: Directory with trained model artifacts.
        output_dir: Directory to write OPE results. Defaults to model_dir/ope.

    Returns:
        Dict with all OPE metrics.
    """
    model_dir = model_dir or settings.model_dir
    output_dir = output_dir or model_dir / "ope"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and calibrator
    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
    with open(model_dir / "cat_maps.json") as f:
        cat_maps = json.load(f)

    calibrator = None
    cal_path = model_dir / "calibrator.pkl"
    if cal_path.exists():
        with open(cal_path, "rb") as f:
            calibrator = pickle.load(f)
        log.info("Loaded calibrator")
    else:
        log.info("No calibrator found, using raw predictions")

    # Load test set
    test_path = model_dir / "test_set.parquet"
    test_df = pd.read_parquet(test_path)
    log.info(f"Test set: {len(test_df):,} rows")

    # Prepare features
    test_with_target = add_target(test_df, TARGET_COL)
    X_test, y_test, _ = prepare_features(
        test_with_target, fit_categories=False, category_maps=cat_maps
    )

    # Predict
    raw_preds = booster.predict(X_test)
    if calibrator is not None:
        preds = calibrator.predict(raw_preds)
    else:
        preds = raw_preds

    # --- Regression Metrics ---
    metrics = regression_metrics(y_test.values, preds)
    log.info(f"Test MAE: {metrics['mae']:.5f}, RMSE: {metrics['rmse']:.5f}")
    log.info(f"Correlation: {metrics['correlation']:.4f}")

    # --- Plots ---
    plot_predicted_vs_actual(y_test.values, preds, output_dir / "pred_vs_actual.png")
    cal_df = plot_calibration_bins(y_test.values, preds, output_dir / "calibration.png")

    importance = dict(
        zip(
            X_test.columns.tolist(),
            booster.feature_importance(importance_type="gain").tolist(),
        )
    )
    plot_feature_importance(importance, output_dir / "feature_importance.png")

    # --- Delay Tier Assignment ---
    pred_series = pd.Series(preds, index=test_with_target.index)
    tiers = assign_tiers(
        pred_series,
        higher=settings.thresholds.higher,
        static_lower=settings.thresholds.static_lower,
        lower_skip=settings.thresholds.lower_skip,
    )

    tier_dist = tiers.value_counts()
    log.info(f"Tier distribution:\n{tier_dist.to_string()}")

    # --- PnL Simulation ---
    sim_df = simulate_policy(test_with_target, tiers)
    summary = policy_summary(sim_df)

    log.info(f"Skip rate: {summary['skip_rate']:.1%}")
    log.info(f"Friction reduced rate: {summary['friction_reduced_rate']:.1%}")
    log.info(f"PnL delta: {summary['pnl_delta']:,.2f} ({summary['pnl_delta_pct']:+.1f}%)")

    # --- HIGHER Tier Review Sample ---
    higher_tickets = sim_df[sim_df["model_tier"] == "HIGHER"].copy()
    higher_tickets["predicted_clv"] = preds[tiers == "HIGHER"]
    if len(higher_tickets) > 100:
        # Stratified sample: 50 with worst predicted CLV, 50 random
        worst_50 = higher_tickets.nsmallest(50, "predicted_clv")
        remaining = higher_tickets.drop(worst_50.index)
        random_50 = remaining.sample(min(50, len(remaining)), random_state=42)
        review_sample = pd.concat([worst_50, random_50])
    else:
        review_sample = higher_tickets

    review_path = output_dir / "higher_review_sample.parquet"
    review_sample.to_parquet(review_path, index=False)
    log.info(f"Saved {len(review_sample)} HIGHER tickets for review → {review_path}")

    # --- Save All Results ---
    all_metrics = {**metrics, **summary}
    with open(output_dir / "ope_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    sim_df.to_parquet(output_dir / "simulation_results.parquet", index=False)

    log.info(f"OPE results saved to {output_dir}")
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Run Offline Policy Evaluation")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    run_ope(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
