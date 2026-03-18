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

TARGET_COL = "odds_after_10"

# Columns that leak future information
LEAKED_COLUMNS = [
    "odds_after_10", "odds_after_30", "odds_after_90",
    "ticket_state", "pnl", "reject_reason",
    "accepted_at", "rejected_at",
]

# IDs/metadata not useful for training
DROP_METADATA = [
    "ticket_id", "bettor_id", "created_at",
    "home_team", "away_team", "tournament", "client_name", "market_params",
]

# LightGBM native categoricals
CATEGORICAL_FEATURES = [
    "market_name", "market_selection", "sport", "odds_bucket",
]
