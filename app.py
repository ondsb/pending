"""Pending Delay Model — unified Streamlit dashboard."""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from pending_delay.config import settings
from pending_delay.features.engineering import engineer_features
from pending_delay.schema import (
    TARGET_COL,
    LEAKED_COLUMNS,
    DROP_METADATA,
    CATEGORICAL_FEATURES,
)

st.set_page_config(page_title="Pending Delay", layout="wide")

PARQUET = str(settings.data_dir / "tickets.parquet")
MODEL_DIR = settings.model_dir
OPE_DIR = MODEL_DIR / "ope"
T = settings.thresholds


# ── Shared helpers ───────────────────────────────────────────────────────────


@st.cache_resource
def get_connection():
    con = duckdb.connect()
    con.sql(f"CREATE VIEW tickets AS SELECT * FROM read_parquet('{PARQUET}')")
    return con


@st.cache_data
def get_total_rows():
    return get_connection().sql("SELECT COUNT(*) FROM tickets").fetchone()[0]


@st.cache_data
def get_column_names():
    return get_connection().sql("SELECT * FROM tickets LIMIT 0").columns


@st.cache_data
def get_distinct_values(col):
    vals = get_connection().sql(f"SELECT DISTINCT {col} FROM tickets WHERE {col} IS NOT NULL ORDER BY {col}").fetchall()
    return [r[0] for r in vals]


def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


con = get_connection()

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_results, tab_review, tab_data, tab_analytics, tab_features = st.tabs(
    [
        "Model Results",
        "HIGHER Review",
        "Data Explorer",
        "Analytics",
        "Feature Analysis",
    ]
)


# =============================================================================
# TAB 1: Model Results
# =============================================================================
with tab_results:
    st.header("Model Results")

    metrics = load_json(MODEL_DIR / "metrics.json")
    ope = load_json(OPE_DIR / "ope_metrics.json")

    if not metrics:
        st.warning("No model found. Run `python build_dataset.py --full` first.")
    else:
        # ── Key metrics ──
        st.subheader("Performance")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Val MAE", f"{metrics['val_mae']:.5f}")
        c2.metric("Val RMSE", f"{metrics['val_rmse']:.5f}")
        c3.metric("Best Iteration", metrics["best_iteration"])
        c4.metric("Train Rows", f"{metrics['n_train']:,}")
        c5.metric("Val Rows", f"{metrics['n_val']:,}")

        if ope:
            st.subheader("Test Set (OPE)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Test MAE", f"{ope['mae']:.5f}")
            c2.metric("Test RMSE", f"{ope['rmse']:.5f}")
            c3.metric("Correlation", f"{ope['correlation']:.4f}")
            c4.metric("Median AE", f"{ope['median_ae']:.5f}")

            st.subheader("Policy Simulation")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Skip Rate", f"{ope['skip_rate']:.1%}")
            c2.metric("Friction Reduced", f"{ope['friction_reduced_rate']:.1%}")
            c3.metric("HIGHER Count", f"{ope['higher_count']:,}")
            c4.metric("PnL Delta", f"{ope['pnl_delta']:,.0f}")
            c5.metric("PnL Delta %", f"{ope['pnl_delta_pct']:+.2f}%")

            # Tier distribution
            tier_data = pd.DataFrame(
                {
                    "Tier": ["SKIP", "LOWER", "STATIC", "HIGHER"],
                    "Count": [
                        ope["skip_count"],
                        ope["lower_count"],
                        ope["static_count"],
                        ope["higher_count"],
                    ],
                }
            )
            tier_data["Pct"] = tier_data["Count"] / tier_data["Count"].sum() * 100
            fig_tier = px.bar(
                tier_data,
                x="Tier",
                y="Count",
                color="Tier",
                color_discrete_map={
                    "SKIP": "#2ecc71",
                    "LOWER": "#3498db",
                    "STATIC": "#f39c12",
                    "HIGHER": "#e74c3c",
                },
                text=tier_data["Pct"].map("{:.1f}%".format),
            )
            fig_tier.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_tier, use_container_width=True)

        # ── Feature importance ──
        st.subheader("Feature Importance (Gain)")
        imp = metrics.get("feature_importance_top20", {})
        if imp:
            imp_df = pd.DataFrame({"Feature": list(imp.keys()), "Gain": list(imp.values())})
            imp_df = imp_df.sort_values("Gain", ascending=True)
            fig_imp = px.bar(imp_df, x="Gain", y="Feature", orientation="h", height=500)
            fig_imp.update_layout(yaxis=dict(dtick=1))
            st.plotly_chart(fig_imp, use_container_width=True)

        # ── Calibration ──
        cal_pre = MODEL_DIR / "calibration_pre.csv"
        cal_post = MODEL_DIR / "calibration_post.csv"
        if cal_pre.exists() and cal_post.exists():
            st.subheader("Calibration")
            col_pre, col_post = st.columns(2)
            with col_pre:
                st.caption("Pre-calibration")
                st.dataframe(pd.read_csv(cal_pre), use_container_width=True, hide_index=True)
            with col_post:
                st.caption("Post-calibration")
                st.dataframe(pd.read_csv(cal_post), use_container_width=True, hide_index=True)

        # ── Saved plots ──
        for img_name, title in [
            ("pred_vs_actual.png", "Predicted vs Actual"),
            ("calibration.png", "Binned Calibration"),
        ]:
            img_path = OPE_DIR / img_name
            if img_path.exists():
                st.image(str(img_path), caption=title, use_container_width=True)


# =============================================================================
# TAB 2: HIGHER Ticket Review
# =============================================================================
with tab_review:
    st.header("HIGHER Tier Review — 100 Selected Tickets")

    review_path = OPE_DIR / "higher_review_sample.parquet"
    if not review_path.exists():
        st.warning("No review sample found. Run the full pipeline first.")
    else:
        df_rev = pd.read_parquet(review_path)

        n_sharp = (df_rev["odds_after_10"] < -0.01).sum()
        n_rejected = (df_rev["ticket_state"] == "rejected").sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tickets", len(df_rev))
        c2.metric("Truly Sharp", f"{n_sharp} ({n_sharp}%)")
        c3.metric("Were Rejected", f"{n_rejected} ({n_rejected}%)")
        c4.metric("Avg Predicted CLV", f"{df_rev['predicted_clv'].mean():.4f}")

        # ── Predicted vs Actual scatter ──
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Predicted vs Actual CLV")
            df_rev["sharp"] = np.where(df_rev["odds_after_10"] < -0.01, "Sharp (< -0.01)", "Not sharp")
            fig_scatter = px.scatter(
                df_rev,
                x="predicted_clv",
                y="odds_after_10",
                color="sharp",
                color_discrete_map={"Sharp (< -0.01)": "crimson", "Not sharp": "steelblue"},
                hover_data=["sport", "market_name", "stake", "ticket_state"],
                opacity=0.7,
            )
            fig_scatter.add_shape(type="line", x0=-0.3, x1=0.3, y0=-0.3, y1=0.3, line=dict(color="gray", dash="dash"))
            fig_scatter.add_hline(y=-0.01, line_dash="dot", line_color="crimson", opacity=0.5)
            fig_scatter.update_layout(height=450)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_b:
            st.subheader("Actual CLV Distribution")
            fig_hist = px.histogram(
                df_rev,
                x="odds_after_10",
                color="sharp",
                nbins=25,
                color_discrete_map={"Sharp (< -0.01)": "crimson", "Not sharp": "steelblue"},
                barmode="overlay",
                opacity=0.7,
            )
            fig_hist.add_vline(x=-0.01, line_dash="dot", line_color="crimson", opacity=0.5)
            fig_hist.update_layout(height=450)
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── By sport + ticket state ──
        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("By Sport")
            sport_stats = (
                df_rev.groupby("sport")
                .agg(
                    count=("odds_after_10", "count"),
                    pct_sharp=("odds_after_10", lambda x: (x < -0.01).mean() * 100),
                    mean_clv=("odds_after_10", "mean"),
                )
                .sort_values("count", ascending=False)
                .reset_index()
            )
            fig_sport = px.bar(
                sport_stats,
                x="sport",
                y="count",
                color="pct_sharp",
                color_continuous_scale="RdYlGn_r",
                text="count",
                labels={"pct_sharp": "% Sharp"},
            )
            fig_sport.update_layout(height=400)
            st.plotly_chart(fig_sport, use_container_width=True)

        with col_d:
            st.subheader("Ticket State")
            state_counts = df_rev["ticket_state"].value_counts().reset_index()
            state_counts.columns = ["state", "count"]
            fig_pie = px.pie(
                state_counts,
                names="state",
                values="count",
                color="state",
                color_discrete_map={
                    "rejected": "#e74c3c",
                    "won": "#2ecc71",
                    "lost": "#3498db",
                    "accepted": "#95a5a6",
                },
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Bettor history vs outcome ──
        col_e, col_f = st.columns(2)

        with col_e:
            st.subheader("Stake vs Actual CLV")
            fig_stake = px.scatter(
                df_rev,
                x="stake",
                y="odds_after_10",
                color="predicted_clv",
                color_continuous_scale="RdYlGn",
                opacity=0.7,
                hover_data=["sport", "market_name", "ticket_state"],
            )
            fig_stake.add_hline(y=-0.01, line_dash="dot", line_color="crimson", opacity=0.5)
            fig_stake.update_layout(height=400)
            st.plotly_chart(fig_stake, use_container_width=True)

        with col_f:
            st.subheader("Bettor History vs Ticket Outcome")
            fig_hist_clv = px.scatter(
                df_rev,
                x="bs_avg_odds_after_10",
                y="odds_after_10",
                color="sharp",
                color_discrete_map={"Sharp (< -0.01)": "crimson", "Not sharp": "steelblue"},
                hover_data=["sport", "bettor_id", "stake"],
                opacity=0.7,
            )
            fig_hist_clv.add_hline(y=-0.01, line_dash="dot", line_color="crimson", opacity=0.5)
            fig_hist_clv.add_vline(x=T.higher, line_dash="dot", line_color="orange", opacity=0.5)
            fig_hist_clv.update_layout(height=400)
            st.plotly_chart(fig_hist_clv, use_container_width=True)

        # ── Detail table ──
        st.subheader("Ticket Details")
        show_cols = [
            "sport",
            "market_name",
            "selection_odds",
            "stake",
            "pending_delay",
            "predicted_clv",
            "odds_after_10",
            "odds_after_30",
            "ticket_state",
            "bs_avg_odds_after_10",
            "bettor_id",
        ]
        available = [c for c in show_cols if c in df_rev.columns]

        sort_col = st.selectbox("Sort by", available, index=available.index("predicted_clv"))
        ascending = st.checkbox("Ascending", value=True)
        st.dataframe(
            df_rev[available].sort_values(sort_col, ascending=ascending).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            height=500,
        )


# =============================================================================
# TAB 3: Data Explorer
# =============================================================================
with tab_data:
    st.header("Data Explorer")

    total_rows = get_total_rows()
    columns = get_column_names()
    st.caption(f"{total_rows:,} total rows | {len(columns)} columns")

    with st.expander("Filters", expanded=True):
        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        ticket_states = get_distinct_values("ticket_state")
        state = fc1.selectbox("Ticket State", ["all"] + ticket_states)
        sports = get_distinct_values("sport")
        sport = fc2.selectbox("Sport", ["all"] + sports)
        reject_reasons = get_distinct_values("reject_reason")
        reject_reason = fc3.selectbox("Reject Reason", ["all"] + reject_reasons)
        min_stake = fc4.number_input("Min Stake", value=0.0, step=1.0)
        max_rows = fc5.number_input("Max Rows", value=5000, min_value=100, max_value=100000, step=1000)

    where = []
    if state != "all":
        where.append(f"ticket_state = '{state}'")
    if sport != "all":
        where.append(f"sport = '{sport}'")
    if reject_reason != "all":
        where.append(f"reject_reason = '{reject_reason}'")
    if min_stake > 0:
        where.append(f"stake >= {min_stake}")
    where_clause = "WHERE " + " AND ".join(where) if where else ""

    with st.expander("Columns"):
        selected_cols = st.multiselect("Show columns", columns, default=list(columns))

    col_list = ", ".join(selected_cols) if selected_cols else "*"
    query = f"SELECT {col_list} FROM tickets {where_clause} LIMIT {int(max_rows)}"
    count_query = f"SELECT COUNT(*) FROM tickets {where_clause}"

    with st.spinner("Querying..."):
        df_browse = con.sql(query).df()
        matched = con.sql(count_query).fetchone()[0]

    st.caption(f"Showing {len(df_browse):,} of {matched:,} matching rows")
    st.dataframe(df_browse, use_container_width=True, height=600)


# =============================================================================
# TAB 4: Analytics
# =============================================================================
with tab_analytics:
    st.header("Analytics")

    overview = con.sql("""
        SELECT
            COUNT(*) AS n,
            COUNT(DISTINCT bettor_id) AS n_bettors,
            COUNT(DISTINCT sport) AS n_sports,
            COUNT(DISTINCT match_id) AS n_matches,
            SUM(CASE WHEN ticket_state = 'won' THEN 1 ELSE 0 END) AS n_won,
            SUM(CASE WHEN ticket_state = 'lost' THEN 1 ELSE 0 END) AS n_lost,
            SUM(CASE WHEN ticket_state = 'rejected' THEN 1 ELSE 0 END) AS n_rejected,
            AVG(odds_after_10) AS avg_clv,
            MEDIAN(odds_after_10) AS med_clv,
            COUNT(odds_after_10) AS n_clv_nonnull,
            SUM(stake) AS total_stake,
            SUM(pnl) AS total_pnl
        FROM tickets
    """).df().iloc[0]

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Tickets", f"{overview['n']:,.0f}")
    m2.metric("Bettors", f"{overview['n_bettors']:,.0f}")
    m3.metric("Sports", f"{overview['n_sports']:,.0f}")
    m4.metric("Matches", f"{overview['n_matches']:,.0f}")
    m5.metric("Total Stake", f"{overview['total_stake']:,.0f}")
    m6.metric("Total PnL", f"{overview['total_pnl']:,.0f}")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Ticket States")
        state_df = con.sql("""
            SELECT ticket_state, COUNT(*) AS n,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct
            FROM tickets GROUP BY ticket_state ORDER BY n DESC
        """).df()
        fig_states = px.bar(state_df, x="ticket_state", y="n", text="pct", color="ticket_state")
        fig_states.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_states, use_container_width=True)

    with col_b:
        st.subheader("Reject Reasons")
        rej_df = con.sql("""
            SELECT reject_reason, COUNT(*) AS n,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct
            FROM tickets WHERE ticket_state = 'rejected' AND reject_reason IS NOT NULL
            GROUP BY reject_reason ORDER BY n DESC
        """).df()
        if len(rej_df) > 0:
            fig_rej = px.bar(rej_df, x="reject_reason", y="n", text="pct", color="reject_reason")
            fig_rej.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_rej, use_container_width=True)

    st.divider()
    st.subheader("Delay Tier Analysis")
    st.caption(
        "Target = `odds_after_10` (line movement 10s post-bet). "
        "Negative = sharp (line moved for bettor). Positive = recreational."
    )

    clv_m1, clv_m2, clv_m3, clv_m4 = st.columns(4)
    clv_m1.metric("Mean CLV", f"{overview['avg_clv']:.5f}")
    clv_m2.metric("Median CLV", f"{overview['med_clv']:.5f}")
    clv_m3.metric("CLV Coverage", f"{overview['n_clv_nonnull']:,.0f} / {overview['n']:,.0f}")
    clv_m4.metric("Rejected", f"{overview['n_rejected']:,.0f}")

    tier_df = con.sql(f"""
        SELECT
            CASE
                WHEN odds_after_10 < {T.higher} THEN '1. HIGHER'
                WHEN odds_after_10 < {T.static_lower} THEN '2. STATIC'
                WHEN odds_after_10 < {T.lower_skip}  THEN '3. LOWER'
                ELSE '4. SKIP'
            END AS tier,
            COUNT(*) AS n,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct,
            ROUND(AVG(odds_after_10), 5) AS avg_clv,
            ROUND(AVG(stake), 2) AS avg_stake,
            ROUND(SUM(pnl), 2) AS total_pnl
        FROM tickets WHERE odds_after_10 IS NOT NULL
        GROUP BY tier ORDER BY tier
    """).df()
    st.dataframe(tier_df, use_container_width=True, hide_index=True)

    st.subheader("CLV by Ticket State")
    clv_state = con.sql("""
        SELECT ticket_state, COUNT(odds_after_10) AS n,
               ROUND(AVG(odds_after_10), 5) AS avg_clv,
               ROUND(MEDIAN(odds_after_10), 5) AS med_clv,
               ROUND(AVG(stake), 2) AS avg_stake,
               ROUND(SUM(pnl), 2) AS total_pnl
        FROM tickets WHERE odds_after_10 IS NOT NULL
        GROUP BY ticket_state ORDER BY n DESC
    """).df()
    st.dataframe(clv_state, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("By Sport")
    sport_df = con.sql("""
        SELECT sport, COUNT(*) AS n, COUNT(DISTINCT bettor_id) AS n_bettors,
               ROUND(AVG(odds_after_10), 5) AS avg_clv,
               ROUND(SUM(stake), 2) AS total_stake,
               ROUND(SUM(pnl), 2) AS total_pnl,
               ROUND(SUM(pnl) / NULLIF(SUM(stake), 0) * 100, 2) AS margin_pct
        FROM tickets GROUP BY sport ORDER BY n DESC
    """).df()
    st.dataframe(sport_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Aggregate Feature Coverage")
    columns = get_column_names()
    agg_cols = [
        c
        for c in columns
        if c.startswith("bs_")
        or c.startswith("total_")
        or c.startswith("avg_rejected")
        or c.startswith("risk_tier")
        or c.startswith("dominant_")
        or c == "mean_stake_size"
        or c == "n_reject_reasons"
    ]
    if agg_cols:
        null_parts = ", ".join(f'ROUND(COUNT({c}) * 100.0 / COUNT(*), 1) AS "{c}"' for c in agg_cols)
        coverage = con.sql(f"SELECT {null_parts} FROM tickets").df()
        coverage_t = coverage.T.reset_index()
        coverage_t.columns = ["feature", "coverage_pct"]
        st.dataframe(coverage_t, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 5: Feature Analysis
# =============================================================================
with tab_features:
    st.header("Feature Analysis")

    st.sidebar.header("Feature Analysis Settings")
    sample_size = st.sidebar.number_input(
        "Sample size", value=200_000, min_value=10_000, max_value=2_000_000, step=50_000
    )
    fa_rounds = st.sidebar.number_input("LightGBM rounds", value=200, min_value=50, max_value=1000, step=50)
    run_perm = st.sidebar.checkbox("Run permutation importance", value=False)

    if st.sidebar.button("Run Feature Analysis", type="primary"):
        import lightgbm as lgb

        excluded = set(LEAKED_COLUMNS + DROP_METADATA + [TARGET_COL])

        with st.spinner(f"Sampling {sample_size:,} rows..."):
            df_fa = con.sql(f"""
                SELECT * FROM tickets
                WHERE {TARGET_COL} IS NOT NULL
                USING SAMPLE {sample_size}
            """).df()
        st.success(f"Loaded {len(df_fa):,} rows")

        df_fa = engineer_features(df_fa)

        target = df_fa[TARGET_COL]
        candidates = [c for c in df_fa.columns if c not in excluded]

        # Null rates
        st.subheader("Feature Coverage")
        null_df = pd.DataFrame(
            {
                "feature": candidates,
                "non_null_pct": [df_fa[c].notna().mean() * 100 for c in candidates],
                "dtype": [str(df_fa[c].dtype) for c in candidates],
            }
        ).sort_values("non_null_pct", ascending=False)
        st.dataframe(null_df, use_container_width=True, hide_index=True)

        # Prepare for LightGBM
        features = df_fa[candidates].copy()
        cat_cols = set(CATEGORICAL_FEATURES)
        for col in cat_cols:
            if col in features.columns:
                features[col] = features[col].fillna("__missing__").astype("category")

        cat_indices = [features.columns.tolist().index(c) for c in cat_cols if c in features.columns]

        dtrain = lgb.Dataset(features, label=target, categorical_feature=cat_indices, free_raw_data=False)

        with st.spinner("Training quick LightGBM..."):
            booster = lgb.train(
                {
                    "objective": "huber",
                    "metric": "mae",
                    "num_leaves": 63,
                    "learning_rate": 0.05,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 5,
                    "min_child_samples": 100,
                    "verbose": -1,
                },
                dtrain,
                num_boost_round=fa_rounds,
            )

        gain = dict(
            zip(
                features.columns,
                booster.feature_importance(importance_type="gain").tolist(),
            )
        )
        split = dict(
            zip(
                features.columns,
                booster.feature_importance(importance_type="split").tolist(),
            )
        )

        imp_df = pd.DataFrame(
            {
                "feature": list(gain.keys()),
                "gain": list(gain.values()),
                "split_count": list(split.values()),
            }
        ).sort_values("gain", ascending=False)
        imp_df["gain_pct"] = imp_df["gain"] / imp_df["gain"].sum() * 100
        imp_df["cumulative_gain_pct"] = imp_df["gain_pct"].cumsum()

        st.subheader("Feature Importance (Gain)")
        fig_gain = px.bar(
            imp_df.head(25),
            x="gain_pct",
            y="feature",
            orientation="h",
            height=500,
            labels={"gain_pct": "Gain %"},
        )
        fig_gain.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_gain, use_container_width=True)
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

        # Correlation
        st.subheader("Correlation with Target")
        numeric_cols = features.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            corr = features[numeric_cols].corrwith(target).abs().sort_values(ascending=False)
            corr_df = pd.DataFrame({"feature": corr.index, "abs_correlation": corr.values})
            fig_corr = px.bar(
                corr_df.head(25),
                x="abs_correlation",
                y="feature",
                orientation="h",
                height=500,
            )
            fig_corr.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_corr, use_container_width=True)

        # Permutation importance
        if run_perm:
            st.subheader("Permutation Importance")
            with st.spinner("Computing..."):
                baseline_preds = booster.predict(features)
                baseline_mae = np.mean(np.abs(baseline_preds - target))
                perm_results = {}
                for col in features.columns:
                    original = features[col].values.copy()
                    features[col] = np.random.permutation(original)
                    perm_mae = np.mean(np.abs(booster.predict(features) - target))
                    perm_results[col] = perm_mae - baseline_mae
                    features[col] = original

            perm_df = pd.DataFrame(
                {
                    "feature": list(perm_results.keys()),
                    "mae_increase": list(perm_results.values()),
                }
            ).sort_values("mae_increase", ascending=False)
            fig_perm = px.bar(
                perm_df.head(25),
                x="mae_increase",
                y="feature",
                orientation="h",
                height=500,
            )
            fig_perm.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_perm, use_container_width=True)

        # Feature selection helper
        st.divider()
        st.subheader("Feature Selection")
        zero_gain = set(imp_df[imp_df["gain"] == 0]["feature"].tolist())
        default_sel = [f for f in imp_df["feature"].tolist() if f not in zero_gain]
        selected = st.multiselect("Features to use", options=imp_df["feature"].tolist(), default=default_sel)
        if selected:
            st.code(f"features = {selected}", language="python")
            st.caption(f"{len(selected)} features selected")
    else:
        st.info("Click **Run Feature Analysis** in the sidebar to start.")
