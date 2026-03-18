"""Tab: Model Comparison -- side-by-side A/B evaluation of two models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pending_delay.config import settings
from pending_delay.evaluation.metrics import regression_metrics
from pending_delay.evaluation.simulate import (
    assign_tiers,
    policy_summary,
    simulate_policy,
)
from pending_delay.ui._loader import (
    compute_predictions,
    discover_model_dirs,
    load_model_artifacts,
)

TIER_ORDER = ["HIGHER", "STATIC", "LOWER", "SKIP"]
TIER_COLORS = {
    "HIGHER": "#e74c3c",
    "STATIC": "#f39c12",
    "LOWER": "#3498db",
    "SKIP": "#2ecc71",
}


def _model_label(path: Path, root: Path) -> str:
    """Create a short display label for a model directory."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _compute_summary(df: pd.DataFrame) -> dict:
    """Compute regression + policy metrics for a prediction DataFrame."""
    reg = regression_metrics(df["actual"].values, df["cal_pred"].values)
    preds = pd.Series(df["cal_pred"].values, index=df.index)
    tiers = assign_tiers(
        preds,
        settings.thresholds.higher,
        settings.thresholds.static_lower,
        settings.thresholds.lower_skip,
    )
    sim = simulate_policy(df, tiers)
    pol = policy_summary(sim)
    return {**reg, **pol, "tiers": tiers}


def _calibration_bins(df: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    """Return binned calibration stats."""
    tmp = df[["cal_pred", "actual"]].copy()
    tmp["bin"] = pd.qcut(tmp["cal_pred"], q=n_bins, duplicates="drop")
    return (
        tmp.groupby("bin", observed=True)
        .agg(
            mean_pred=("cal_pred", "mean"),
            mean_actual=("actual", "mean"),
            n=("actual", "count"),
        )
        .reset_index()
    )


def render(models_root: Path) -> None:
    """Render the Model Comparison tab."""
    dirs = discover_model_dirs(models_root)

    if len(dirs) < 2:
        st.info(
            "Model comparison requires at least two trained models.  "
            f"Found {len(dirs)} model(s) under `{models_root}`."
        )
        return

    # ------------------------------------------------------------------
    # Model selectors
    # ------------------------------------------------------------------
    st.subheader("Model Comparison (A / B)")
    st.caption("Compare two model versions side by side on their held-out test sets.")

    labels = [_model_label(d, models_root) for d in dirs]
    dir_map = dict(zip(labels, dirs))

    sc1, sc2 = st.columns(2)
    label_a = sc1.selectbox("Model A", labels, index=0, key="cmp_a")
    label_b = sc2.selectbox(
        "Model B",
        labels,
        index=min(1, len(labels) - 1),
        key="cmp_b",
    )

    if label_a == label_b:
        st.warning("Select two different models to compare.")
        return

    dir_a = str(dir_map[label_a])
    dir_b = str(dir_map[label_b])

    # ------------------------------------------------------------------
    # Load predictions
    # ------------------------------------------------------------------
    with st.spinner("Loading models and computing predictions..."):
        df_a = compute_predictions(dir_a)
        df_b = compute_predictions(dir_b)
        arts_a = load_model_artifacts(dir_a)
        arts_b = load_model_artifacts(dir_b)
        summ_a = _compute_summary(df_a)
        summ_b = _compute_summary(df_b)

    # Test set size warning
    if abs(len(df_a) - len(df_b)) / max(len(df_a), len(df_b)) > 0.05:
        st.warning(
            f"Test set sizes differ significantly: Model A has {len(df_a):,} rows, "
            f"Model B has {len(df_b):,} rows.  Metrics may not be directly comparable "
            f"if the models were trained on different data splits."
        )

    # ------------------------------------------------------------------
    # 1. Side-by-side metrics
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Regression Metrics")

    metric_keys = [
        ("MAE", "mae", True),  # (label, key, lower_is_better)
        ("RMSE", "rmse", True),
        ("Median AE", "median_ae", True),
        ("Pearson r", "correlation", False),
    ]

    cols = st.columns(len(metric_keys))
    for col, (label, key, lower_better) in zip(cols, metric_keys):
        val_a = summ_a[key]
        val_b = summ_b[key]
        diff = val_b - val_a
        # Positive delta = B is higher.  For lower-is-better, positive = worse.
        improved = (diff < 0) if lower_better else (diff > 0)
        arrow = "improved" if improved else "worsened"
        col.metric(
            label,
            f"A: {val_a:.5f}",
            delta=f"B: {val_b:.5f} ({diff:+.5f})",
            delta_color="normal" if improved else "inverse",
        )

    # ------------------------------------------------------------------
    # 2. Policy metrics side by side
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Policy Impact")

    pol_keys = [
        ("Skip Rate", "skip_rate", "{:.1%}", False),
        ("Friction Reduced", "friction_reduced_rate", "{:.1%}", False),
        ("PnL Delta", "pnl_delta", "{:,.0f}", False),
        ("PnL Delta %", "pnl_delta_pct", "{:+.1f}%", False),
    ]

    pcols = st.columns(len(pol_keys))
    for col, (label, key, fmt, lower_better) in zip(pcols, pol_keys):
        va = summ_a[key]
        vb = summ_b[key]
        diff = vb - va
        improved = (diff < 0) if lower_better else (diff > 0)
        col.metric(
            label,
            f"A: {fmt.format(va)}",
            delta=f"B: {fmt.format(vb)}",
            delta_color="normal" if improved else "inverse",
        )

    # ------------------------------------------------------------------
    # 3. Calibration overlay
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Calibration Comparison")

    cal_a = _calibration_bins(df_a)
    cal_b = _calibration_bins(df_b)

    fig_cal = go.Figure()
    fig_cal.add_trace(
        go.Scatter(
            x=cal_a["mean_pred"],
            y=cal_a["mean_actual"],
            mode="lines+markers",
            name=f"Model A ({label_a})",
            line=dict(color="#3498db"),
        )
    )
    fig_cal.add_trace(
        go.Scatter(
            x=cal_b["mean_pred"],
            y=cal_b["mean_actual"],
            mode="lines+markers",
            name=f"Model B ({label_b})",
            line=dict(color="#e74c3c"),
        )
    )
    # Perfect line
    all_preds = list(cal_a["mean_pred"]) + list(cal_b["mean_pred"])
    lim = [min(all_preds), max(all_preds)]
    fig_cal.add_trace(
        go.Scatter(
            x=lim,
            y=lim,
            mode="lines",
            name="Perfect",
            line=dict(dash="dash", color="grey"),
        )
    )
    fig_cal.update_layout(
        xaxis_title="Mean Predicted CLV",
        yaxis_title="Mean Actual CLV",
        height=450,
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    # ------------------------------------------------------------------
    # 4. Tier distribution comparison
    # ------------------------------------------------------------------
    st.divider()
    col_tier, col_imp = st.columns(2)

    with col_tier:
        st.subheader("Tier Distribution Shift")
        tiers_a = summ_a["tiers"]
        tiers_b = summ_b["tiers"]
        ct_a = tiers_a.value_counts().reindex(TIER_ORDER, fill_value=0)
        ct_b = tiers_b.value_counts().reindex(TIER_ORDER, fill_value=0)
        pct_a = ct_a / ct_a.sum() * 100
        pct_b = ct_b / ct_b.sum() * 100

        tier_cmp = pd.DataFrame(
            {
                "Tier": TIER_ORDER * 2,
                "Pct": list(pct_a.values) + list(pct_b.values),
                "Model": [f"A ({label_a})"] * 4 + [f"B ({label_b})"] * 4,
            }
        )
        fig_tier = px.bar(
            tier_cmp,
            x="Tier",
            y="Pct",
            color="Model",
            barmode="group",
            text=tier_cmp["Pct"].apply(lambda p: f"{p:.1f}%"),
            labels={"Pct": "% of tickets"},
            color_discrete_sequence=["#3498db", "#e74c3c"],
        )
        fig_tier.update_layout(height=400)
        st.plotly_chart(fig_tier, use_container_width=True)

    # ------------------------------------------------------------------
    # 5. Feature importance diff
    # ------------------------------------------------------------------
    with col_imp:
        st.subheader("Feature Importance Shift")
        imp_a: dict = arts_a["metrics"].get("feature_importance_top20", {})
        imp_b: dict = arts_b["metrics"].get("feature_importance_top20", {})

        if imp_a and imp_b:
            all_features = sorted(set(list(imp_a.keys()) + list(imp_b.keys())))
            imp_diff = []
            for f in all_features:
                va = imp_a.get(f, 0)
                vb = imp_b.get(f, 0)
                # Normalize to relative importance
                max_a = max(imp_a.values()) if imp_a else 1
                max_b = max(imp_b.values()) if imp_b else 1
                rel_a = va / max_a * 100 if max_a > 0 else 0
                rel_b = vb / max_b * 100 if max_b > 0 else 0
                imp_diff.append(
                    {
                        "feature": f,
                        "diff": rel_b - rel_a,
                        "A_rel": rel_a,
                        "B_rel": rel_b,
                    }
                )

            diff_df = pd.DataFrame(imp_diff).sort_values("diff")

            fig_imp = px.bar(
                diff_df,
                x="diff",
                y="feature",
                orientation="h",
                color="diff",
                color_continuous_scale=["#e74c3c", "#f7f7f7", "#2ecc71"],
                color_continuous_midpoint=0,
                labels={"diff": "Relative Importance (B - A)", "feature": "Feature"},
                hover_data=["A_rel", "B_rel"],
            )
            fig_imp.update_layout(height=max(350, len(diff_df) * 22))
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance data not available for both models.")

    # ------------------------------------------------------------------
    # 6. Verdict
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Summary")

    verdicts = []
    mae_diff = summ_b["mae"] - summ_a["mae"]
    if mae_diff < -0.00001:
        verdicts.append(f"Model B has **lower MAE** ({mae_diff:+.5f})")
    elif mae_diff > 0.00001:
        verdicts.append(f"Model B has **higher MAE** ({mae_diff:+.5f})")
    else:
        verdicts.append("MAE is roughly equivalent between models")

    pnl_diff = summ_b["pnl_delta"] - summ_a["pnl_delta"]
    if pnl_diff > 0:
        verdicts.append(f"Model B produces a **better PnL delta** ({pnl_diff:+,.0f})")
    elif pnl_diff < 0:
        verdicts.append(f"Model B produces a **worse PnL delta** ({pnl_diff:+,.0f})")

    skip_diff = summ_b["skip_rate"] - summ_a["skip_rate"]
    if abs(skip_diff) > 0.001:
        direction = "higher" if skip_diff > 0 else "lower"
        verdicts.append(f"Model B has **{direction} skip rate** ({skip_diff:+.1%})")

    for v in verdicts:
        st.markdown(f"- {v}")
