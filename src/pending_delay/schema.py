"""Data schemas and column definitions."""

import pyarrow as pa

# Raw ticket schema from Athena CSV export
TICKET_SCHEMA = pa.schema([
    ("ticket_id", pa.string()),
    ("created_at", pa.string()),
    ("accepted_at", pa.string()),
    ("rejected_at", pa.string()),
    ("bettor_id", pa.string()),
    ("pending_delay", pa.int32()),
    ("client_id", pa.int32()),
    ("cashout_stake", pa.float64()),
    ("reject_reason", pa.string()),
    ("ticket_state", pa.string()),
    ("match_id", pa.int32()),
    ("home_team", pa.string()),
    ("away_team", pa.string()),
    ("stake", pa.float64()),
    ("pnl", pa.float64()),
    ("selection_odds", pa.float64()),
    ("market_name", pa.string()),
    ("market_type_id", pa.int32()),
    ("market_params", pa.string()),
    ("market_selection", pa.string()),
    ("client_name", pa.string()),
    ("bos", pa.float64()),
    ("oaf", pa.float64()),
    ("ots_risk_tier_id", pa.int64()),
    ("sport_id", pa.int64()),
    ("sport", pa.string()),
    ("tournament_id", pa.int64()),
    ("tournament", pa.string()),
    ("odds_after_10", pa.float64()),
    ("odds_after_30", pa.float64()),
    ("odds_after_90", pa.float64()),
])

# Columns that leak future information — must be dropped before training
LEAKED_COLUMNS = [
    "odds_after_10",   # per-ticket post-hoc line movement (TARGET source)
    "odds_after_30",   # per-ticket post-hoc
    "odds_after_90",   # per-ticket post-hoc
    "ticket_state",    # outcome
    "pnl",             # outcome-derived
    "reject_reason",   # post-decision (populated after delay decision)
    "accepted_at",     # post-decision timestamp
    "rejected_at",     # post-decision timestamp
]

# Columns that are IDs/metadata — not useful for training
DROP_METADATA = [
    "ticket_id",
    "bettor_id",
    "created_at",  # used for splitting only, not as feature
    "home_team",
    "away_team",
    "tournament",
    "client_name",
    "market_params",
]

# Features available at inference time (ticket-level)
TICKET_FEATURES = [
    "stake",
    "selection_odds",
    "cashout_stake",
    "market_type_id",
    "market_name",
    "market_selection",
    "sport",
    "sport_id",
    "tournament_id",
    "match_id",
    "client_id",
    "bos",
    "oaf",
    "ots_risk_tier_id",
    "pending_delay",
]

# Bettor-level aggregate features (historical, pre-computed)
AGGREGATE_FEATURES = [
    "bs_pnl",
    "bs_margin",
    "bs_stake",
    "bs_avg_odds_after_10",
    "bs_avg_odds_after_30",
    "bs_avg_odds_after_90",
    "bs_rejected_stake",
    "bs_rejected_pnl",
    "total_rejected_stake",
    "total_rejected_pnl",
    "n_reject_reasons",
    "avg_rejected_odds_after_10",
    "avg_rejected_odds_after_30",
    "avg_rejected_odds_after_90",
    "dominant_risk_tier",
    "risk_tier_total_pnl",
    "risk_tier_total_volume",
    "risk_tier_avg_margin",
    "mean_stake_size",
]

# Engineered features (computed in feature pipeline)
ENGINEERED_FEATURES = [
    "stake_ratio",
    "odds_bucket",
]

# LightGBM native categoricals
CATEGORICAL_FEATURES = [
    "market_name",
    "market_selection",
    "sport",
    "odds_bucket",
]

# All training features in order
ALL_FEATURES = TICKET_FEATURES + AGGREGATE_FEATURES + ENGINEERED_FEATURES

# Target column name
TARGET_COL = "odds_after_10"
