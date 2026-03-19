"""Pending Delay Model -- validation dashboard.

Streamlit app for validating pre-trained LightGBM models that predict
bettor CLV (odds_after_10) and assign pending-delay tiers.

Launch:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from pending_delay.config import settings
from pending_delay.ui._loader import discover_model_dirs

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Pending Delay", layout="wide")

PARQUET = settings.data_dir / "tickets.parquet"
MODELS_ROOT = settings.model_dir
T = settings.thresholds


# ── Shared helpers (DuckDB for data tabs) ────────────────────────────────────


@st.cache_resource
def get_connection():
    pq = str(PARQUET)
    con = duckdb.connect()
    con.sql(f"CREATE VIEW tickets AS SELECT * FROM read_parquet('{pq}')")
    return con


@st.cache_data
def get_total_rows():
    return get_connection().sql("SELECT COUNT(*) FROM tickets").fetchone()[0]


@st.cache_data
def get_column_names():
    return get_connection().sql("SELECT * FROM tickets LIMIT 0").columns


@st.cache_data
def get_distinct_values(col):
    rows = (
        get_connection()
        .sql(
            f"SELECT DISTINCT {col} FROM tickets WHERE {col} IS NOT NULL ORDER BY {col}"
        )
        .fetchall()
    )
    return [r[0] for r in rows]


def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Sidebar: model picker ────────────────────────────────────────────────────

model_dirs = discover_model_dirs(MODELS_ROOT)
selected_model_dir: str | None = None

with st.sidebar:
    st.header("Model")

    if model_dirs:
        labels = {str(d.relative_to(MODELS_ROOT)): str(d) for d in model_dirs}
        choice = st.selectbox(
            "Model directory",
            list(labels.keys()),
            key="sidebar_model",
        )
        selected_model_dir = labels[choice]
        st.caption(f"`{selected_model_dir}`")
    else:
        st.info(
            "No models found.\n\n"
            "Validate an external model first:\n"
            "```\n"
            "python validate_model.py \\\n"
            "  --model path/to/model.txt \\\n"
            "  --data data/tickets.parquet\n"
            "```"
        )

has_data = PARQUET.exists()
has_model = selected_model_dir is not None


# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_titles = ["Data Explorer", "Analytics"]
if has_model:
    tab_titles = [
        "Validation",
        "Tier Simulator",
        "Segments",
        "Review Queue",
    ] + tab_titles
if len(model_dirs) >= 2:
    tab_titles.append("Model Comparison")

tabs = st.tabs(tab_titles)

# Work out which tab object corresponds to which name.  The model-specific
# tabs only exist when a model is selected, so we track by position.
_idx = 0


def _next_tab():
    global _idx
    t = tabs[_idx]
    _idx += 1
    return t


# ── Model-specific tabs (only when a model is selected) ─────────────────────

if has_model:
    # --- Validation ---
    with _next_tab():
        from pending_delay.ui.tab_validation import render as render_validation

        render_validation(selected_model_dir)

    # --- Tier Simulator ---
    with _next_tab():
        from pending_delay.ui.tab_tier_simulator import render as render_simulator

        render_simulator(selected_model_dir)

    # --- Segment Analysis ---
    with _next_tab():
        from pending_delay.ui.tab_segments import render as render_segments

        render_segments(selected_model_dir)

    # --- Review Queue ---
    with _next_tab():
        from pending_delay.ui.tab_review import render as render_review

        render_review(selected_model_dir)


# ── Data Explorer (always available) ────────────────────────────────────────

con = get_connection()

with _next_tab():
    st.header("Data Explorer")

    if not has_data:
        st.error(
            f"No data found at `{PARQUET}`. "
            "Run `python convert_to_parquet.py --merge` first."
        )
    else:
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
            max_rows = fc5.number_input(
                "Max Rows", value=5000, min_value=100, max_value=100000, step=1000
            )

        where: list[str] = []
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
            selected_cols = st.multiselect(
                "Show columns", columns, default=list(columns)
            )

        col_list = ", ".join(selected_cols) if selected_cols else "*"
        query = f"SELECT {col_list} FROM tickets {where_clause} LIMIT {int(max_rows)}"
        count_query = f"SELECT COUNT(*) FROM tickets {where_clause}"

        with st.spinner("Querying..."):
            df_browse = con.sql(query).df()
            matched = con.sql(count_query).fetchone()[0]

        st.caption(f"Showing {len(df_browse):,} of {matched:,} matching rows")
        st.dataframe(df_browse, use_container_width=True, height=600)


# ── Analytics (always available) ─────────────────────────────────────────────

with _next_tab():
    st.header("Analytics")

    if not has_data:
        st.error("No data file found.")
    else:
        overview = (
            con.sql("""
            SELECT
                COUNT(*) AS n,
                COUNT(DISTINCT bettor_id) AS n_bettors,
                COUNT(DISTINCT sport) AS n_sports,
                COUNT(DISTINCT match_id) AS n_matches,
                SUM(CASE WHEN ticket_state = 'won' THEN 1 ELSE 0 END) AS n_won,
                SUM(CASE WHEN ticket_state = 'lost' THEN 1 ELSE 0 END) AS n_lost,
                SUM(CASE WHEN ticket_state = 'rejected' THEN 1 ELSE 0 END)
                    AS n_rejected,
                AVG(odds_after_10) AS avg_clv,
                MEDIAN(odds_after_10) AS med_clv,
                COUNT(odds_after_10) AS n_clv_nonnull,
                SUM(stake) AS total_stake,
                SUM(pnl) AS total_pnl
            FROM tickets
        """)
            .df()
            .iloc[0]
        )

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
            fig_states = px.bar(
                state_df,
                x="ticket_state",
                y="n",
                text="pct",
                color="ticket_state",
            )
            fig_states.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_states, use_container_width=True)

        with col_b:
            st.subheader("Reject Reasons")
            rej_df = con.sql("""
                SELECT reject_reason, COUNT(*) AS n,
                       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct
                FROM tickets
                WHERE ticket_state = 'rejected' AND reject_reason IS NOT NULL
                GROUP BY reject_reason ORDER BY n DESC
            """).df()
            if len(rej_df) > 0:
                fig_rej = px.bar(
                    rej_df,
                    x="reject_reason",
                    y="n",
                    text="pct",
                    color="reject_reason",
                )
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
        clv_m3.metric(
            "CLV Coverage",
            f"{overview['n_clv_nonnull']:,.0f} / {overview['n']:,.0f}",
        )
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
            SELECT sport, COUNT(*) AS n,
                   COUNT(DISTINCT bettor_id) AS n_bettors,
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
            null_parts = ", ".join(
                f'ROUND(COUNT({c}) * 100.0 / COUNT(*), 1) AS "{c}"' for c in agg_cols
            )
            coverage = con.sql(f"SELECT {null_parts} FROM tickets").df()
            coverage_t = coverage.T.reset_index()
            coverage_t.columns = ["feature", "coverage_pct"]
            st.dataframe(coverage_t, use_container_width=True, hide_index=True)


# ── Model Comparison (needs 2+ models) ──────────────────────────────────────

if len(model_dirs) >= 2:
    with _next_tab():
        from pending_delay.ui.tab_comparison import render as render_comparison

        render_comparison(MODELS_ROOT)
