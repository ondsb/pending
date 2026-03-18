"""Tab: Review Queue -- browse HIGHER-tier tickets for manual inspection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from pending_delay.config import settings
from pending_delay.evaluation.simulate import assign_tiers
from pending_delay.ui._loader import compute_predictions

REVIEW_COLUMNS = [
    "cal_pred",
    "actual",
    "stake",
    "sport",
    "market_name",
    "market_selection",
    "selection_odds",
    "client_id",
    "ots_risk_tier_id",
    "pending_delay",
    "bs_pnl",
    "bs_margin",
    "bs_stake",
    "dominant_risk_tier",
    "pnl",
    "ticket_state",
]


def render(model_dir: str) -> None:
    """Render the Review Queue tab."""
    df = compute_predictions(model_dir)

    # Try loading pre-computed review sample first
    review_path = Path(model_dir) / "ope" / "higher_review_sample.parquet"
    precomputed = review_path.exists()

    # Assign tiers on the full test set for live stats
    preds = pd.Series(df["cal_pred"].values, index=df.index)
    tiers = assign_tiers(
        preds,
        settings.thresholds.higher,
        settings.thresholds.static_lower,
        settings.thresholds.lower_skip,
    )

    higher_df = df[tiers == "HIGHER"].copy()

    # ------------------------------------------------------------------
    # Summary metrics
    # ------------------------------------------------------------------
    st.subheader("HIGHER Tier Review")
    st.caption(
        "Tickets the model would assign the most punitive delay.  "
        "Review these to sanity-check the model's sharpness detection."
    )

    n_higher = len(higher_df)
    n_total = len(df)

    if n_higher == 0:
        st.info("No tickets assigned to the HIGHER tier with current thresholds.")
        return

    mean_pred = higher_df["cal_pred"].mean()
    mean_actual = higher_df["actual"].mean()

    # "Hit rate": % where actual CLV also falls below the HIGHER threshold
    hit_rate = (higher_df["actual"] < settings.thresholds.higher).mean()
    # "False positive rate": % where actual CLV >= 0 (bettor was recreational)
    fp_rate = (higher_df["actual"] >= 0).mean()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        "HIGHER Tickets",
        f"{n_higher:,}",
        delta=f"{n_higher / n_total * 100:.1f}% of total",
    )
    m2.metric("Mean Predicted CLV", f"{mean_pred:.5f}")
    m3.metric("Mean Actual CLV", f"{mean_actual:.5f}")
    m4.metric(
        "Hit Rate",
        f"{hit_rate:.1%}",
        help="% where actual CLV is also below HIGHER threshold",
    )
    m5.metric(
        "False Positive Rate",
        f"{fp_rate:.1%}",
        help="% where actual CLV >= 0 (recreational misclassified as sharp)",
    )

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------
    st.divider()

    with st.expander("Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)

        # Sport filter
        sports = (
            ["all"] + sorted(higher_df["sport"].dropna().unique().tolist())
            if "sport" in higher_df.columns
            else ["all"]
        )
        sport_filter = fc1.selectbox("Sport", sports, key="review_sport")

        # Min stake
        min_stake = fc2.number_input(
            "Min Stake", value=0.0, step=1.0, key="review_min_stake"
        )

        # Sort by
        sort_options = {
            "Predicted CLV (worst first)": ("cal_pred", True),
            "Actual CLV (worst first)": ("actual", True),
            "Stake (highest first)": ("stake", False),
        }
        sort_choice = fc3.selectbox(
            "Sort by", list(sort_options.keys()), key="review_sort"
        )

    # Apply filters
    filtered = higher_df.copy()
    if sport_filter != "all":
        filtered = filtered[filtered["sport"] == sport_filter]
    if min_stake > 0:
        filtered = filtered[filtered["stake"] >= min_stake]

    sort_col, ascending = sort_options[sort_choice]
    filtered = filtered.sort_values(sort_col, ascending=ascending)

    # ------------------------------------------------------------------
    # Review table
    # ------------------------------------------------------------------
    st.divider()

    available_cols = [c for c in REVIEW_COLUMNS if c in filtered.columns]
    display = filtered[available_cols].copy()

    # Rename for clarity
    rename_map = {
        "cal_pred": "Predicted CLV",
        "actual": "Actual CLV",
    }
    display = display.rename(
        columns={k: v for k, v in rename_map.items() if k in display.columns}
    )

    st.caption(f"Showing {len(display):,} HIGHER-tier tickets")

    max_rows = st.number_input(
        "Max rows to display",
        value=500,
        min_value=50,
        max_value=5000,
        step=100,
        key="review_max_rows",
    )
    st.dataframe(
        display.head(max_rows),
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    if precomputed:
        st.caption(
            f"Pre-computed review sample also available at `{review_path}` "
            f"(50 worst + 50 random from OPE run)."
        )
