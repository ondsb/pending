"""Counterfactual PnL simulation for delay tier policies.

Given a test set with known outcomes (ticket_state, pnl, odds movements),
simulate what would happen under different delay tier assignments.
"""

import logging

import numpy as np
import pandas as pd

from pending_delay.features.target import classify_toxicity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def assign_tiers(
    predictions: pd.Series,
    higher: float = -0.02,
    static_lower: float = -0.005,
    lower_skip: float = 0.005,
) -> pd.Series:
    """Assign delay tiers from continuous predictions."""
    return classify_toxicity(predictions, higher, static_lower, lower_skip)


def simulate_policy(
    test_df: pd.DataFrame,
    tiers: pd.Series,
) -> pd.DataFrame:
    """Simulate outcomes under the model-assigned delay tiers.

    For each ticket in the test set:
    - SKIP/LOWER: ticket is accepted faster → use actual pnl as outcome
      (conservative: we assume same outcome regardless of delay reduction)
    - STATIC: no change → use actual pnl
    - HIGHER: punitive delay → some fraction would be rejected by bettor
      (model those with large negative odds_after_10 as deterred)

    The simulation is conservative: we don't assume SKIP/LOWER changes outcomes,
    we only count the friction reduction benefit. The real value is in correctly
    identifying which tickets to expedite vs delay.
    """
    result = test_df.copy()
    result["model_tier"] = tiers.values

    # Factual: all tickets under static policy use their actual PnL
    result["factual_pnl"] = result["pnl"].fillna(0)

    # Counterfactual: under model policy
    # For SKIP/LOWER/STATIC: same PnL (conservative assumption)
    result["counterfactual_pnl"] = result["factual_pnl"].copy()

    # For HIGHER: tickets with sharp line movement (negative odds_after_10)
    # would face longer delay → some would be deterred/rejected
    # Conservative: only count as "averted" if odds moved significantly against book
    higher_mask = result["model_tier"] == "HIGHER"
    sharp_and_higher = higher_mask & (result["odds_after_10"] < -0.01)

    # These sharp bets under HIGHER delay would likely be rejected/withdrawn
    # The book avoids the negative PnL from these
    result.loc[sharp_and_higher, "counterfactual_pnl"] = 0.0

    return result


def policy_summary(sim_df: pd.DataFrame) -> dict:
    """Compute summary statistics from simulation results."""
    n_total = len(sim_df)
    tier_counts = sim_df["model_tier"].value_counts()

    skip_count = tier_counts.get("SKIP", 0)
    lower_count = tier_counts.get("LOWER", 0)
    static_count = tier_counts.get("STATIC", 0)
    higher_count = tier_counts.get("HIGHER", 0)

    factual_pnl = sim_df["factual_pnl"].sum()
    counterfactual_pnl = sim_df["counterfactual_pnl"].sum()

    return {
        "n_total": n_total,
        "skip_count": int(skip_count),
        "skip_rate": float(skip_count / n_total) if n_total > 0 else 0,
        "lower_count": int(lower_count),
        "lower_rate": float(lower_count / n_total) if n_total > 0 else 0,
        "friction_reduced_rate": float((skip_count + lower_count) / n_total) if n_total > 0 else 0,
        "static_count": int(static_count),
        "higher_count": int(higher_count),
        "factual_pnl": float(factual_pnl),
        "counterfactual_pnl": float(counterfactual_pnl),
        "pnl_delta": float(counterfactual_pnl - factual_pnl),
        "pnl_delta_pct": float(
            (counterfactual_pnl - factual_pnl) / abs(factual_pnl) * 100
        ) if factual_pnl != 0 else 0,
    }
