"""Tab: Segment Analysis -- per-segment model performance drilldown."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
from pending_delay.ui._loader import compute_predictions

TIER_ORDER = ["HIGHER", "STATIC", "LOWER", "SKIP"]

# Columns that make sense as segment dimensions
SEGMENT_DIMS = [
    "sport",
    "market_name",
    "ots_risk_tier_id",
    "client_id",
    "dominant_risk_tier",
]


def _segment_metrics(
    group: pd.DataFrame,
    higher: float,
    static_lower: float,
    lower_skip: float,
) -> pd.Series:
    """Compute validation + policy metrics for a single segment group."""
    errors = group["cal_pred"].values - group["actual"].values
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    n = len(group)
    corr = (
        float(np.corrcoef(group["actual"], group["cal_pred"])[0, 1]) if n > 2 else 0.0
    )

    preds = pd.Series(group["cal_pred"].values, index=group.index)
    tiers = assign_tiers(preds, higher, static_lower, lower_skip)
    sim = simulate_policy(group, tiers)
    summ = policy_summary(sim)

    return pd.Series(
        {
            "count": n,
            "mae": mae,
            "rmse": rmse,
            "correlation": corr,
            "skip_rate": summ["skip_rate"],
            "higher_rate": summ.get("higher_count", 0) / n if n > 0 else 0,
            "pnl_delta": summ["pnl_delta"],
            "factual_pnl": summ["factual_pnl"],
            "counterfactual_pnl": summ["counterfactual_pnl"],
        }
    )


def render(model_dir: str) -> None:
    """Render the Segment Analysis tab."""
    df = compute_predictions(model_dir)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    st.subheader("Segment Analysis")
    st.caption(
        "Break down model performance by data segments to identify where the model underperforms."
    )

    ctrl1, ctrl2 = st.columns([1, 1])

    # Only offer dimensions that exist in the data
    available_dims = [d for d in SEGMENT_DIMS if d in df.columns]
    # Add odds_bucket if it can be derived
    if "selection_odds" in df.columns:
        available_dims.append("odds_bucket")

    segment_dim = ctrl1.selectbox(
        "Segment dimension",
        available_dims,
        key="seg_dim",
    )
    min_count = ctrl2.number_input(
        "Min tickets per segment",
        value=100,
        min_value=1,
        step=50,
        key="seg_min_count",
    )

    # Independent threshold sliders for this tab
    with st.expander("Threshold overrides (independent from Tier Simulator)"):
        sc1, sc2, sc3 = st.columns(3)
        higher = sc1.slider(
            "HIGHER",
            -0.10,
            0.00,
            settings.thresholds.higher,
            step=0.001,
            format="%.3f",
            key="seg_higher",
        )
        static_lower = sc2.slider(
            "STATIC/LOWER",
            -0.05,
            0.01,
            settings.thresholds.static_lower,
            step=0.001,
            format="%.3f",
            key="seg_static",
        )
        lower_skip = sc3.slider(
            "LOWER/SKIP",
            -0.01,
            0.05,
            settings.thresholds.lower_skip,
            step=0.001,
            format="%.3f",
            key="seg_skip",
        )

    if not (higher < static_lower < lower_skip):
        st.error("Thresholds must be ordered: HIGHER < STATIC/LOWER < LOWER/SKIP")
        st.stop()

    # ------------------------------------------------------------------
    # Compute odds_bucket on the fly if needed
    # ------------------------------------------------------------------
    work_df = df.copy()
    if segment_dim == "odds_bucket" and "odds_bucket" not in work_df.columns:
        work_df["odds_bucket"] = pd.cut(
            work_df["selection_odds"],
            bins=[0, 1.3, 1.7, 2.2, 3.5, float("inf")],
            labels=["heavy_fav", "slight_fav", "even", "underdog", "longshot"],
        ).astype(str)

    # ------------------------------------------------------------------
    # Per-segment metrics
    # ------------------------------------------------------------------
    with st.spinner("Computing per-segment metrics..."):
        seg_results = (
            work_df.groupby(segment_dim, observed=True)
            .apply(
                lambda g: _segment_metrics(g, higher, static_lower, lower_skip),
                include_groups=False,
            )
            .reset_index()
        )

    seg_results = seg_results[seg_results["count"] >= min_count].sort_values(
        "count",
        ascending=False,
    )

    if seg_results.empty:
        st.warning("No segments meet the minimum count threshold.")
        return

    overall_mae = float(np.mean(np.abs(df["cal_pred"].values - df["actual"].values)))

    # ------------------------------------------------------------------
    # 1. Segment table
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Per-Segment Metrics")

    display = seg_results.copy()
    display["mae"] = display["mae"].round(5)
    display["rmse"] = display["rmse"].round(5)
    display["correlation"] = display["correlation"].round(4)
    display["skip_rate"] = (display["skip_rate"] * 100).round(1).astype(str) + "%"
    display["higher_rate"] = (display["higher_rate"] * 100).round(1).astype(str) + "%"
    display["pnl_delta"] = display["pnl_delta"].round(0).astype(int)
    display["factual_pnl"] = display["factual_pnl"].round(0).astype(int)
    display["counterfactual_pnl"] = display["counterfactual_pnl"].round(0).astype(int)
    display["count"] = display["count"].astype(int)

    st.dataframe(display, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # 2. Worst segments callout
    # ------------------------------------------------------------------
    high_mae = seg_results[seg_results["mae"] > overall_mae * 1.5]
    if not high_mae.empty:
        names = ", ".join(str(v) for v in high_mae[segment_dim].tolist()[:5])
        st.warning(f"Segments with MAE > 1.5x overall ({overall_mae:.5f}): **{names}**")

    neg_pnl = seg_results.nsmallest(3, "pnl_delta")
    if not neg_pnl.empty and (neg_pnl["pnl_delta"] < 0).any():
        names = ", ".join(str(v) for v in neg_pnl[segment_dim].tolist())
        st.warning(f"Segments with most negative PnL delta: **{names}**")

    # ------------------------------------------------------------------
    # 3. Visualizations
    # ------------------------------------------------------------------
    st.divider()
    col_mae, col_pnl = st.columns(2)

    with col_mae:
        st.subheader("MAE by Segment")
        top_n = seg_results.nlargest(20, "count")
        fig_mae = px.bar(
            top_n.sort_values("mae"),
            x="mae",
            y=segment_dim,
            orientation="h",
            labels={"mae": "MAE", segment_dim: segment_dim.replace("_", " ").title()},
        )
        fig_mae.add_vline(
            x=overall_mae,
            line_dash="dash",
            line_color="red",
            annotation_text=f"overall={overall_mae:.5f}",
        )
        fig_mae.update_layout(height=max(350, len(top_n) * 25))
        st.plotly_chart(fig_mae, use_container_width=True)

    with col_pnl:
        st.subheader("PnL Delta by Segment")
        fig_pnl = px.bar(
            top_n.sort_values("pnl_delta"),
            x="pnl_delta",
            y=segment_dim,
            orientation="h",
            color="pnl_delta",
            color_continuous_scale=["#e74c3c", "#f7f7f7", "#2ecc71"],
            color_continuous_midpoint=0,
            labels={
                "pnl_delta": "PnL Delta",
                segment_dim: segment_dim.replace("_", " ").title(),
            },
        )
        fig_pnl.update_layout(height=max(350, len(top_n) * 25))
        st.plotly_chart(fig_pnl, use_container_width=True)

    # ------------------------------------------------------------------
    # 4. Segment detail expander
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Segment Detail")
    selected_seg = st.selectbox(
        f"Select a {segment_dim} to drill down",
        seg_results[segment_dim].tolist(),
        key="seg_detail_select",
    )

    if selected_seg is not None:
        seg_data = work_df[work_df[segment_dim] == selected_seg]
        if len(seg_data) == 0:
            st.info("No data for this segment.")
            return

        dc1, dc2 = st.columns(2)

        with dc1:
            st.markdown(f"**Calibration -- {selected_seg}**")
            cal_seg = seg_data[["cal_pred", "actual"]].copy()
            cal_seg["bin"] = pd.qcut(
                cal_seg["cal_pred"],
                q=min(10, len(cal_seg) // 20 + 1),
                duplicates="drop",
            )
            cal_agg = (
                cal_seg.groupby("bin", observed=True)
                .agg(
                    mean_pred=("cal_pred", "mean"),
                    mean_actual=("actual", "mean"),
                    n=("actual", "count"),
                )
                .reset_index()
            )
            fig_seg_cal = go.Figure()
            fig_seg_cal.add_trace(
                go.Bar(
                    x=cal_agg["mean_pred"],
                    y=cal_agg["mean_actual"],
                    name="Actual",
                    marker_color="#3498db",
                    opacity=0.7,
                )
            )
            fig_seg_cal.add_trace(
                go.Scatter(
                    x=cal_agg["mean_pred"],
                    y=cal_agg["mean_pred"],
                    mode="lines",
                    name="Perfect",
                    line=dict(dash="dash", color="grey"),
                )
            )
            fig_seg_cal.update_layout(
                height=350,
                xaxis_title="Mean Predicted",
                yaxis_title="Mean Actual",
            )
            st.plotly_chart(fig_seg_cal, use_container_width=True)

        with dc2:
            st.markdown(f"**Tier Distribution -- {selected_seg}**")
            seg_preds = pd.Series(seg_data["cal_pred"].values, index=seg_data.index)
            seg_tiers = assign_tiers(seg_preds, higher, static_lower, lower_skip)
            tier_ct = seg_tiers.value_counts().reindex(TIER_ORDER, fill_value=0)
            fig_seg_tier = px.pie(
                names=tier_ct.index,
                values=tier_ct.values,
                color=tier_ct.index,
                color_discrete_map={
                    "HIGHER": "#e74c3c",
                    "STATIC": "#f39c12",
                    "LOWER": "#3498db",
                    "SKIP": "#2ecc71",
                },
            )
            fig_seg_tier.update_layout(height=350)
            st.plotly_chart(fig_seg_tier, use_container_width=True)
