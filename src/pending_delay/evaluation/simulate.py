"""Counterfactual PnL simulation for delay tier policies."""

import pandas as pd

from pending_delay.features.target import classify_toxicity

# Re-export under the name used by other modules (create_dummy_model, etc.)
assign_tiers = classify_toxicity


def simulate_policy(test_df: pd.DataFrame, tiers: pd.Series) -> pd.DataFrame:
    """Simulate outcomes under the model-assigned delay tiers.

    - SKIP/LOWER/STATIC: same PnL (conservative — we don't assume faster
      acceptance changes the outcome)
    - HIGHER: tickets with sharp line movement (odds_after_10 < -0.01)
      would face longer delay and likely be rejected/withdrawn by the bettor,
      so the book averts that negative PnL.
    """
    result = test_df.copy()
    result["model_tier"] = tiers.values

    result["factual_pnl"] = result["pnl"].fillna(0)
    result["counterfactual_pnl"] = result["factual_pnl"].copy()

    # Sharp bets under HIGHER delay: bettor likely withdraws
    sharp_and_higher = (result["model_tier"] == "HIGHER") & (
        result["odds_after_10"] < -0.01
    )
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
        "friction_reduced_rate": float((skip_count + lower_count) / n_total)
        if n_total > 0
        else 0,
        "static_count": int(static_count),
        "higher_count": int(higher_count),
        "factual_pnl": float(factual_pnl),
        "counterfactual_pnl": float(counterfactual_pnl),
        "pnl_delta": float(counterfactual_pnl - factual_pnl),
        "pnl_delta_pct": float(
            (counterfactual_pnl - factual_pnl) / abs(factual_pnl) * 100
        )
        if factual_pnl != 0
        else 0,
    }
