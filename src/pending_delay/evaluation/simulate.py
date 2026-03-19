"""Counterfactual PnL simulation for binary delay policy (SKIP vs PENDING)."""

import pandas as pd


def simulate_policy(test_df: pd.DataFrame, tiers: pd.Series) -> pd.DataFrame:
    """Simulate outcomes under binary delay policy.

    Status quo: all bets go through pending delay (some get rejected).
    Counterfactual: SKIP bets bypass delay → rejected SKIP bets now go through.

    For rejected bets moved to SKIP:
      - We estimate their PnL using the yield (pnl/stake) of accepted bets
        in the same tier, since we don't observe the actual outcome.
    For PENDING bets: no change from status quo.
    """
    result = test_df.copy()
    result["model_tier"] = tiers.values
    result["factual_pnl"] = result["pnl"].fillna(0)
    result["counterfactual_pnl"] = result["factual_pnl"].copy()

    # Estimate yield from accepted (settled) SKIP bets
    skip_mask = result["model_tier"] == "SKIP"
    settled = skip_mask & result["ticket_state"].isin(["won", "lost"])
    skip_yield = (
        result.loc[settled, "factual_pnl"].sum()
        / result.loc[settled, "stake"].sum()
    ) if result.loc[settled, "stake"].sum() != 0 else 0.0

    # Rejected SKIP bets would now go through → estimate their PnL
    skip_rejected = skip_mask & (result["ticket_state"] == "rejected")
    result.loc[skip_rejected, "counterfactual_pnl"] = (
        result.loc[skip_rejected, "stake"] * skip_yield
    )

    return result


def policy_summary(sim_df: pd.DataFrame) -> dict:
    """Compute summary statistics from simulation results."""
    n_total = len(sim_df)
    tier_counts = sim_df["model_tier"].value_counts()

    skip_count = tier_counts.get("SKIP", 0)
    pending_count = tier_counts.get("PENDING", 0)

    skip_mask = sim_df["model_tier"] == "SKIP"
    skip_rejected = skip_mask & (sim_df["ticket_state"] == "rejected")

    factual_pnl = sim_df["factual_pnl"].sum()
    counterfactual_pnl = sim_df["counterfactual_pnl"].sum()

    return {
        "n_total": n_total,
        "skip_count": int(skip_count),
        "skip_rate": float(skip_count / n_total) if n_total > 0 else 0,
        "pending_count": int(pending_count),
        "pending_rate": float(pending_count / n_total) if n_total > 0 else 0,
        "skip_rejected_count": int(skip_rejected.sum()),
        "factual_pnl": float(factual_pnl),
        "counterfactual_pnl": float(counterfactual_pnl),
        "pnl_delta": float(counterfactual_pnl - factual_pnl),
        "pnl_delta_pct": float(
            (counterfactual_pnl - factual_pnl) / abs(factual_pnl) * 100
        ) if factual_pnl != 0 else 0,
    }
