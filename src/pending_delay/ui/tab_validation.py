"""Tab: Model Validation -- regression diagnostics, calibration, feature importance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pending_delay.config import settings
from pending_delay.evaluation.metrics import regression_metrics
from pending_delay.ui._loader import compute_predictions, load_model_artifacts

# ---------------------------------------------------------------------------
# Tier colours (consistent across all tabs)
# ---------------------------------------------------------------------------
TIER_COLORS = {
    "HIGHER": "#e74c3c",
    "STATIC": "#f39c12",
    "LOWER": "#3498db",
    "SKIP": "#2ecc71",
}


def _tier_for_pred(pred: float) -> str:
    if pred < settings.thresholds.higher:
        return "HIGHER"
    if pred < settings.thresholds.static_lower:
        return "STATIC"
    if pred < settings.thresholds.lower_skip:
        return "LOWER"
    return "SKIP"


# ---------------------------------------------------------------------------
# Public render
# ---------------------------------------------------------------------------


def render(model_dir: str) -> None:
    """Render the Model Validation tab."""
    arts = load_model_artifacts(model_dir)
    df = compute_predictions(model_dir)
    metrics = regression_metrics(df["actual"].values, df["cal_pred"].values)
    train_metrics: dict = arts["metrics"]

    # ------------------------------------------------------------------
    # 1. Metric cards
    # ------------------------------------------------------------------
    st.subheader("Regression Metrics")
    st.caption(f"Evaluated on {len(df):,} held-out test rows")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("MAE", f"{metrics['mae']:.5f}")
    c2.metric("RMSE", f"{metrics['rmse']:.5f}")
    c3.metric("Median AE", f"{metrics['median_ae']:.5f}")
    c4.metric("Pearson r", f"{metrics['correlation']:.4f}")
    bias = metrics["mean_pred"] - metrics["mean_actual"]
    c5.metric("Bias", f"{bias:+.5f}")
    c6.metric(
        "Best Iter",
        str(train_metrics.get("best_iteration", "?")),
    )

    # ------------------------------------------------------------------
    # 2. Predicted vs Actual scatter
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Predicted vs Actual CLV")

    sample_n = min(15_000, len(df))
    plot_df = df.sample(sample_n, random_state=42).copy()
    plot_df["tier"] = plot_df["cal_pred"].apply(_tier_for_pred)

    fig_scatter = px.scatter(
        plot_df,
        x="actual",
        y="cal_pred",
        color="tier",
        color_discrete_map=TIER_COLORS,
        opacity=0.25,
        labels={"actual": "Actual CLV (odds_after_10)", "cal_pred": "Predicted CLV"},
        category_orders={"tier": ["HIGHER", "STATIC", "LOWER", "SKIP"]},
        hover_data=["stake", "sport"],
    )
    # Perfect calibration line
    lims = [
        min(plot_df["actual"].min(), plot_df["cal_pred"].min()),
        max(plot_df["actual"].max(), plot_df["cal_pred"].max()),
    ]
    fig_scatter.add_trace(
        go.Scatter(
            x=lims,
            y=lims,
            mode="lines",
            line=dict(dash="dash", color="grey"),
            name="Perfect",
            showlegend=True,
        )
    )
    fig_scatter.update_layout(height=550)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ------------------------------------------------------------------
    # 3. Residual distribution
    # ------------------------------------------------------------------
    st.divider()
    col_res, col_err = st.columns(2)

    with col_res:
        st.subheader("Residual Distribution")
        residuals = df["cal_pred"].values - df["actual"].values
        fig_resid = px.histogram(
            x=residuals,
            nbins=100,
            labels={"x": "Residual (predicted - actual)", "y": "Count"},
        )
        mean_r = float(np.mean(residuals))
        med_r = float(np.median(residuals))
        fig_resid.add_vline(
            x=mean_r,
            line_dash="dash",
            line_color="red",
            annotation_text=f"mean={mean_r:.4f}",
        )
        fig_resid.add_vline(
            x=med_r,
            line_dash="dot",
            line_color="blue",
            annotation_text=f"median={med_r:.4f}",
        )
        fig_resid.update_layout(height=400)
        st.plotly_chart(fig_resid, use_container_width=True)

    # ------------------------------------------------------------------
    # 4. Error by prediction range
    # ------------------------------------------------------------------
    with col_err:
        st.subheader("MAE by Prediction Range")
        binned = df[["cal_pred", "actual"]].copy()
        binned["bin"] = pd.qcut(binned["cal_pred"], q=20, duplicates="drop")
        bin_stats = (
            binned.groupby("bin", observed=True)
            .apply(
                lambda g: pd.Series(
                    {
                        "mae": np.mean(np.abs(g["cal_pred"] - g["actual"])),
                        "count": len(g),
                        "bin_center": g["cal_pred"].mean(),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        fig_bin_mae = px.bar(
            bin_stats,
            x="bin_center",
            y="mae",
            labels={"bin_center": "Prediction Bin Center", "mae": "MAE"},
            hover_data=["count"],
        )
        overall_mae = metrics["mae"]
        fig_bin_mae.add_hline(
            y=overall_mae,
            line_dash="dash",
            line_color="red",
            annotation_text=f"overall MAE={overall_mae:.4f}",
        )
        fig_bin_mae.update_layout(height=400)
        st.plotly_chart(fig_bin_mae, use_container_width=True)

    # ------------------------------------------------------------------
    # 5. Calibration analysis
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Calibration Analysis")

    n_bins_cal = st.slider(
        "Number of calibration bins",
        10,
        50,
        20,
        key="val_cal_bins",
    )

    cal_df = df[["cal_pred", "actual"]].copy()
    cal_df["bin"] = pd.qcut(cal_df["cal_pred"], q=n_bins_cal, duplicates="drop")
    cal_stats = (
        cal_df.groupby("bin", observed=True)
        .agg(
            n=("actual", "count"),
            mean_pred=("cal_pred", "mean"),
            mean_actual=("actual", "mean"),
        )
        .reset_index()
    )

    fig_cal = go.Figure()
    fig_cal.add_trace(
        go.Bar(
            x=cal_stats["mean_pred"],
            y=cal_stats["mean_actual"],
            name="Actual Mean",
            marker_color="#3498db",
            opacity=0.7,
        )
    )
    fig_cal.add_trace(
        go.Scatter(
            x=cal_stats["mean_pred"],
            y=cal_stats["mean_pred"],
            mode="lines",
            name="Perfect",
            line=dict(dash="dash", color="grey"),
        )
    )
    fig_cal.update_layout(
        xaxis_title="Mean Predicted CLV (bin)",
        yaxis_title="Mean Actual CLV",
        height=400,
        barmode="group",
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    # Also show the raw numbers
    with st.expander("Calibration bin details"):
        cal_stats["abs_error"] = np.abs(
            cal_stats["mean_pred"] - cal_stats["mean_actual"]
        )
        st.dataframe(
            cal_stats[["mean_pred", "mean_actual", "abs_error", "n"]].round(5),
            use_container_width=True,
            hide_index=True,
        )

    # ------------------------------------------------------------------
    # 6. Feature importance
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Feature Importance (Gain)")

    importance: dict = train_metrics.get("feature_importance_top20", {})
    if importance:
        imp_df = pd.DataFrame(
            list(importance.items()), columns=["feature", "importance"]
        ).sort_values("importance", ascending=True)
        fig_imp = px.bar(
            imp_df,
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Importance (Gain)", "feature": "Feature"},
        )
        fig_imp.update_layout(height=max(350, len(imp_df) * 22))
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("No feature importance data found in model metrics.")

    # ------------------------------------------------------------------
    # 7. Distribution overlay
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Prediction vs Actual Distribution")

    fig_dist = go.Figure()
    fig_dist.add_trace(
        go.Histogram(
            x=df["actual"].values,
            nbinsx=100,
            name="Actual CLV",
            opacity=0.5,
            marker_color="#3498db",
        )
    )
    fig_dist.add_trace(
        go.Histogram(
            x=df["cal_pred"].values,
            nbinsx=100,
            name="Predicted CLV",
            opacity=0.5,
            marker_color="#e74c3c",
        )
    )
    fig_dist.update_layout(
        barmode="overlay",
        xaxis_title="CLV (odds_after_10)",
        yaxis_title="Count",
        height=400,
    )
    st.plotly_chart(fig_dist, use_container_width=True)
