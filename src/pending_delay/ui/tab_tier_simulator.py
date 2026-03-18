"""Tab: Tier Simulator -- interactive threshold tuning with live PnL simulation."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pending_delay.config import settings
from pending_delay.evaluation.simulate import (
    assign_tiers,
    policy_summary,
    simulate_policy,
)
from pending_delay.features.target import add_target
from pending_delay.schema import TARGET_COL
from pending_delay.ui._loader import compute_predictions

# Default thresholds from config
_DEF_HIGHER = settings.thresholds.higher
_DEF_STATIC = settings.thresholds.static_lower
_DEF_SKIP = settings.thresholds.lower_skip

TIER_COLORS = {
    "HIGHER": "#e74c3c",
    "STATIC": "#f39c12",
    "LOWER": "#3498db",
    "SKIP": "#2ecc71",
}
TIER_ORDER = ["HIGHER", "STATIC", "LOWER", "SKIP"]


def _compute_default_summary(df: pd.DataFrame) -> dict:
    """Run the simulation with default thresholds for delta comparison."""
    preds = pd.Series(df["cal_pred"].values, index=df.index)
    tiers = assign_tiers(preds, _DEF_HIGHER, _DEF_STATIC, _DEF_SKIP)
    sim = simulate_policy(df, tiers)
    return policy_summary(sim)


def render(model_dir: str) -> None:
    """Render the Tier Simulator tab."""
    df = compute_predictions(model_dir)

    # ------------------------------------------------------------------
    # Threshold controls
    # ------------------------------------------------------------------
    st.subheader("Threshold Controls")
    st.caption(
        "Adjust the CLV thresholds that map continuous predictions to delay tiers.  "
        "All metrics below update live."
    )

    tc1, tc2, tc3, tc4 = st.columns([1, 1, 1, 0.5])
    higher = tc1.slider(
        "HIGHER threshold",
        min_value=-0.10,
        max_value=0.00,
        value=_DEF_HIGHER,
        step=0.001,
        format="%.3f",
        help="Predictions below this -> HIGHER (punitive delay)",
        key="sim_higher",
    )
    static_lower = tc2.slider(
        "STATIC/LOWER boundary",
        min_value=-0.05,
        max_value=0.01,
        value=_DEF_STATIC,
        step=0.001,
        format="%.3f",
        help="Predictions between HIGHER and this -> STATIC",
        key="sim_static",
    )
    lower_skip = tc3.slider(
        "LOWER/SKIP boundary",
        min_value=-0.01,
        max_value=0.05,
        value=_DEF_SKIP,
        step=0.001,
        format="%.3f",
        help="Predictions above this -> SKIP (bypass delay)",
        key="sim_skip",
    )
    if tc4.button("Reset defaults", key="sim_reset"):
        st.session_state["sim_higher"] = _DEF_HIGHER
        st.session_state["sim_static"] = _DEF_STATIC
        st.session_state["sim_skip"] = _DEF_SKIP
        st.rerun()

    # Validate ordering
    if not (higher < static_lower < lower_skip):
        st.error("Thresholds must be ordered: HIGHER < STATIC/LOWER < LOWER/SKIP")
        st.stop()

    # ------------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------------
    preds = pd.Series(df["cal_pred"].values, index=df.index)
    tiers = assign_tiers(preds, higher, static_lower, lower_skip)
    sim_df = simulate_policy(df, tiers)
    summary = policy_summary(sim_df)
    default_summary = _compute_default_summary(df)

    # ------------------------------------------------------------------
    # 1. Headline metrics
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Policy Impact")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Skip Rate",
        f"{summary['skip_rate']:.1%}",
        delta=f"{summary['skip_rate'] - default_summary['skip_rate']:+.1%}"
        if higher != _DEF_HIGHER
        or static_lower != _DEF_STATIC
        or lower_skip != _DEF_SKIP
        else None,
    )
    m2.metric(
        "Friction Reduced",
        f"{summary['friction_reduced_rate']:.1%}",
        delta=f"{summary['friction_reduced_rate'] - default_summary['friction_reduced_rate']:+.1%}"
        if higher != _DEF_HIGHER
        or static_lower != _DEF_STATIC
        or lower_skip != _DEF_SKIP
        else None,
    )
    m3.metric(
        "PnL Delta",
        f"{summary['pnl_delta']:,.0f}",
        delta=f"{summary['pnl_delta'] - default_summary['pnl_delta']:+,.0f}"
        if higher != _DEF_HIGHER
        or static_lower != _DEF_STATIC
        or lower_skip != _DEF_SKIP
        else None,
    )
    m4.metric(
        "PnL Delta %",
        f"{summary['pnl_delta_pct']:+.1f}%",
        delta=f"{summary['pnl_delta_pct'] - default_summary['pnl_delta_pct']:+.1f}pp"
        if higher != _DEF_HIGHER
        or static_lower != _DEF_STATIC
        or lower_skip != _DEF_SKIP
        else None,
    )

    # ------------------------------------------------------------------
    # 2. Tier distribution
    # ------------------------------------------------------------------
    st.divider()
    col_dist, col_pnl = st.columns(2)

    with col_dist:
        st.subheader("Tier Distribution")
        tier_counts = tiers.value_counts().reindex(TIER_ORDER, fill_value=0)
        tier_pcts = tier_counts / tier_counts.sum() * 100
        dist_df = pd.DataFrame(
            {
                "tier": TIER_ORDER,
                "count": [tier_counts[t] for t in TIER_ORDER],
                "pct": [tier_pcts[t] for t in TIER_ORDER],
            }
        )
        fig_dist = px.bar(
            dist_df,
            x="tier",
            y="count",
            color="tier",
            color_discrete_map=TIER_COLORS,
            text=dist_df["pct"].apply(lambda p: f"{p:.1f}%"),
            labels={"count": "Ticket Count", "tier": "Delay Tier"},
        )
        fig_dist.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

    # ------------------------------------------------------------------
    # 3. PnL comparison
    # ------------------------------------------------------------------
    with col_pnl:
        st.subheader("PnL Comparison")
        pnl_df = pd.DataFrame(
            {
                "Policy": ["Factual (status quo)", "Counterfactual (model)"],
                "PnL": [summary["factual_pnl"], summary["counterfactual_pnl"]],
            }
        )
        fig_pnl = px.bar(
            pnl_df,
            x="Policy",
            y="PnL",
            color="Policy",
            color_discrete_sequence=["#95a5a6", "#2ecc71"],
            text=pnl_df["PnL"].apply(lambda v: f"{v:,.0f}"),
        )
        fig_pnl.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_pnl, use_container_width=True)

    # ------------------------------------------------------------------
    # 4. Per-tier breakdown table
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Per-Tier Breakdown")

    tier_rows = []
    for tier in TIER_ORDER:
        mask = sim_df["model_tier"] == tier
        subset = sim_df[mask]
        if len(subset) == 0:
            continue
        tier_rows.append(
            {
                "Tier": tier,
                "Count": len(subset),
                "%": f"{len(subset) / len(sim_df) * 100:.1f}",
                "Mean Actual CLV": round(subset["odds_after_10"].mean(), 5)
                if "odds_after_10" in subset.columns
                else None,
                "Mean Predicted CLV": round(
                    df.loc[mask.index[mask], "cal_pred"].mean(), 5
                ),
                "Avg Stake": round(subset["stake"].mean(), 2)
                if "stake" in subset.columns
                else None,
                "Factual PnL": round(subset["factual_pnl"].sum(), 2),
                "Counterfactual PnL": round(subset["counterfactual_pnl"].sum(), 2),
            }
        )

    st.dataframe(pd.DataFrame(tier_rows), use_container_width=True, hide_index=True)
