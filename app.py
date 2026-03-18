"""Streamlit browser for pending delay ticket data."""

import streamlit as st
import duckdb
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Pending Delay Explorer", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
LOCAL_PARQUET = DATA_DIR / "tickets.parquet"

# Tier thresholds (from pending_delay.config defaults)
T_HIGHER = -0.02
T_STATIC = -0.005
T_LOWER = 0.005


@st.cache_resource
def get_connection():
    return duckdb.connect()


@st.cache_data
def get_total_rows():
    con = get_connection()
    return con.sql(f"SELECT COUNT(*) FROM read_parquet('{LOCAL_PARQUET}')").fetchone()[0]


@st.cache_data
def get_column_names():
    con = get_connection()
    return con.sql(f"SELECT * FROM read_parquet('{LOCAL_PARQUET}') LIMIT 0").columns


@st.cache_data
def get_distinct_values(col):
    con = get_connection()
    vals = con.sql(
        f"SELECT DISTINCT {col} FROM read_parquet('{LOCAL_PARQUET}') WHERE {col} IS NOT NULL ORDER BY {col}"
    ).fetchall()
    return [r[0] for r in vals]


con = get_connection()

st.title("Pending Delay Explorer")

if not LOCAL_PARQUET.exists():
    st.error(f"No data found at `{LOCAL_PARQUET}`. Run `python convert_to_parquet.py --merge` first.")
    st.stop()

total_rows = get_total_rows()
columns = get_column_names()

tab_data, tab_analytics = st.tabs(["Dataset", "Analytics"])

# =============================================================================
# TAB 1: Dataset browser
# =============================================================================
with tab_data:
    st.caption(f"{total_rows:,} total rows | {len(columns)} columns")

    # Filters
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

    # Build query
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

    # Column selector
    with st.expander("Columns"):
        selected_cols = st.multiselect("Show columns", columns, default=list(columns))

    col_list = ", ".join(selected_cols) if selected_cols else "*"
    query = f"SELECT {col_list} FROM read_parquet('{LOCAL_PARQUET}') {where_clause} LIMIT {int(max_rows)}"
    count_query = f"SELECT COUNT(*) FROM read_parquet('{LOCAL_PARQUET}') {where_clause}"

    with st.spinner("Querying..."):
        df = con.sql(query).df()
        matched = con.sql(count_query).fetchone()[0]

    st.caption(f"Showing {len(df):,} of {matched:,} matching rows")
    st.dataframe(df, use_container_width=True, height=600)

# =============================================================================
# TAB 2: Analytics
# =============================================================================
with tab_analytics:

    # --- Overview metrics (computed via DuckDB for speed) ---
    overview = con.sql(f"""
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
        FROM read_parquet('{LOCAL_PARQUET}')
    """).df().iloc[0]

    st.subheader("Overview")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Tickets", f"{overview['n']:,.0f}")
    m2.metric("Bettors", f"{overview['n_bettors']:,.0f}")
    m3.metric("Sports", f"{overview['n_sports']:,.0f}")
    m4.metric("Matches", f"{overview['n_matches']:,.0f}")
    m5.metric("Total Stake", f"{overview['total_stake']:,.0f}")
    m6.metric("Total PnL", f"{overview['total_pnl']:,.0f}")

    # --- Ticket state distribution ---
    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Ticket States")
        state_df = con.sql(f"""
            SELECT ticket_state, COUNT(*) AS n,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct
            FROM read_parquet('{LOCAL_PARQUET}')
            GROUP BY ticket_state ORDER BY n DESC
        """).df()
        st.dataframe(state_df, use_container_width=True, hide_index=True)
        st.bar_chart(state_df.set_index("ticket_state")["n"])

    with col_b:
        st.subheader("Reject Reasons")
        rej_df = con.sql(f"""
            SELECT reject_reason, COUNT(*) AS n,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct
            FROM read_parquet('{LOCAL_PARQUET}')
            WHERE ticket_state = 'rejected' AND reject_reason IS NOT NULL
            GROUP BY reject_reason ORDER BY n DESC
        """).df()
        if len(rej_df) > 0:
            st.dataframe(rej_df, use_container_width=True, hide_index=True)
            st.bar_chart(rej_df.set_index("reject_reason")["n"])
        else:
            st.info("No rejected tickets found.")

    # --- CLV / Delay Tier Analysis ---
    st.divider()
    st.subheader("Delay Tier Analysis")
    st.caption(
        "Target = `odds_after_10` (line movement 10s post-bet). "
        "Negative = sharp (line moved for bettor). Positive = recreational."
    )

    clv_m1, clv_m2, clv_m3, clv_m4 = st.columns(4)
    clv_m1.metric("Mean CLV", f"{overview['avg_clv']:.5f}")
    clv_m2.metric("Median CLV", f"{overview['med_clv']:.5f}")
    # stddev via pandas to avoid DuckDB STDDEV_SAMP overflow
    _std = con.sql(f"SELECT odds_after_10 FROM read_parquet('{LOCAL_PARQUET}') WHERE odds_after_10 IS NOT NULL LIMIT 500000").df()["odds_after_10"].std()
    clv_m3.metric("Std Dev", f"{_std:.5f}" if pd.notna(_std) else "—")
    clv_m4.metric("Coverage", f"{overview['n_clv_nonnull']:,.0f} / {overview['n']:,.0f}")

    tier_df = con.sql(f"""
        SELECT
            CASE
                WHEN odds_after_10 < {T_HIGHER} THEN '1. HIGHER (punitive delay)'
                WHEN odds_after_10 < {T_STATIC} THEN '2. STATIC (keep current)'
                WHEN odds_after_10 < {T_LOWER}  THEN '3. LOWER (reduce delay)'
                ELSE '4. SKIP (accept immediately)'
            END AS tier,
            COUNT(*) AS n,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct,
            ROUND(AVG(odds_after_10), 5) AS avg_clv,
            ROUND(AVG(stake), 2) AS avg_stake,
            ROUND(SUM(pnl), 2) AS total_pnl
        FROM read_parquet('{LOCAL_PARQUET}')
        WHERE odds_after_10 IS NOT NULL
        GROUP BY tier ORDER BY tier
    """).df()
    st.dataframe(tier_df, use_container_width=True, hide_index=True)

    # CLV by ticket state
    st.subheader("CLV by Ticket State")
    clv_state = con.sql(f"""
        SELECT
            ticket_state,
            COUNT(odds_after_10) AS n,
            ROUND(AVG(odds_after_10), 5) AS avg_clv,
            ROUND(MEDIAN(odds_after_10), 5) AS med_clv,
            ROUND(AVG(stake), 2) AS avg_stake,
            ROUND(SUM(pnl), 2) AS total_pnl
        FROM read_parquet('{LOCAL_PARQUET}')
        WHERE odds_after_10 IS NOT NULL
        GROUP BY ticket_state ORDER BY n DESC
    """).df()
    st.dataframe(clv_state, use_container_width=True, hide_index=True)

    # --- Sport breakdown ---
    st.divider()
    st.subheader("By Sport")
    sport_df = con.sql(f"""
        SELECT
            sport,
            COUNT(*) AS n,
            COUNT(DISTINCT bettor_id) AS n_bettors,
            ROUND(AVG(odds_after_10), 5) AS avg_clv,
            ROUND(SUM(stake), 2) AS total_stake,
            ROUND(SUM(pnl), 2) AS total_pnl,
            ROUND(SUM(pnl) / NULLIF(SUM(stake), 0) * 100, 2) AS margin_pct
        FROM read_parquet('{LOCAL_PARQUET}')
        GROUP BY sport ORDER BY n DESC
    """).df()
    st.dataframe(sport_df, use_container_width=True, hide_index=True)

    # --- Aggregate feature coverage ---
    st.divider()
    st.subheader("Aggregate Feature Coverage")
    agg_cols = [c for c in columns if c.startswith("bs_") or c.startswith("total_")
                or c.startswith("avg_rejected") or c.startswith("risk_tier")
                or c.startswith("dominant_") or c == "mean_stake_size" or c == "n_reject_reasons"]
    if agg_cols:
        null_parts = ", ".join(
            f"ROUND(COUNT({c}) * 100.0 / COUNT(*), 1) AS \"{c}\"" for c in agg_cols
        )
        coverage = con.sql(f"""
            SELECT {null_parts}
            FROM read_parquet('{LOCAL_PARQUET}')
        """).df()
        # Transpose for readability
        coverage_t = coverage.T.reset_index()
        coverage_t.columns = ["feature", "coverage_pct"]
        st.dataframe(coverage_t, use_container_width=True, hide_index=True)
    else:
        st.warning("No aggregate features found — did you run --merge?")
