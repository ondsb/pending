"""Generate a dummy model with synthetic data for UI testing.

Creates realistic-looking fake data, trains a tiny LightGBM model on it,
fits a calibrator, runs OPE, and saves all artifacts to models/dummy/.

Usage:
    python create_dummy_model.py
"""

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from pending_delay.config import settings
from pending_delay.evaluation.metrics import regression_metrics
from pending_delay.evaluation.simulate import (
    assign_tiers,
    policy_summary,
    simulate_policy,
)
from pending_delay.features.engineering import prepare_features, get_categorical_indices
from pending_delay.features.target import add_target
from pending_delay.schema import (
    AGGREGATE_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    TICKET_FEATURES,
)

N_ROWS = 20_000
RNG = np.random.default_rng(42)
MODEL_DIR = settings.model_dir / "dummy"


def generate_synthetic_data(n: int = N_ROWS) -> pd.DataFrame:
    """Generate a synthetic dataset that mimics the real ticket data."""

    # --- Bettor profiles (some sharp, mostly recreational) ---
    n_bettors = n // 10
    bettor_ids = [f"bettor_{i:05d}" for i in range(n_bettors)]
    # ~20% of bettors are "sharp"
    bettor_sharp = RNG.random(n_bettors) < 0.20

    # Assign each ticket a bettor
    bettor_idx = RNG.integers(0, n_bettors, size=n)

    sports = [
        "soccer",
        "basketball",
        "tennis",
        "esports_dota2",
        "esports_csgo",
        "hockey",
        "volleyball",
    ]
    markets = ["1x2", "over_under", "handicap", "both_to_score", "correct_score"]
    selections = ["home", "away", "draw", "over", "under", "yes", "no"]
    clients = list(range(1, 8))

    # --- Ticket-level features ---
    sport_choices = RNG.choice(sports, size=n)
    sport_id_map = {s: i + 1 for i, s in enumerate(sports)}

    stake = np.abs(RNG.lognormal(mean=2.0, sigma=1.2, size=n))
    selection_odds = np.clip(
        RNG.lognormal(mean=0.5, sigma=0.6, size=n) + 1.0, 1.01, 50.0
    )

    # --- Target: odds_after_10 (CLV) ---
    # Sharp bettors get negative CLV on average, recreational get ~0
    is_sharp_ticket = bettor_sharp[bettor_idx]
    base_clv = np.where(
        is_sharp_ticket,
        RNG.normal(-0.03, 0.02, size=n),  # sharp: negative CLV
        RNG.normal(0.002, 0.015, size=n),  # recreational: slightly positive
    )
    odds_after_10 = base_clv + RNG.normal(0, 0.005, size=n)  # noise

    # --- Outcomes ---
    # Simplified: PnL correlates loosely with CLV
    win_prob = np.clip(0.45 + odds_after_10 * 2, 0.1, 0.9)
    won = RNG.random(n) < win_prob
    pnl = np.where(won, stake * (selection_odds - 1), -stake)

    # ~5% rejected
    rejected_mask = RNG.random(n) < 0.05
    ticket_state = np.where(rejected_mask, "rejected", np.where(won, "won", "lost"))
    reject_reasons = ["max_stake", "odds_change", "market_closed", "risk_management"]
    reject_reason = np.where(
        rejected_mask,
        RNG.choice(reject_reasons, size=n),
        None,
    )

    # --- Timestamps (temporal ordering) ---
    base_ts = pd.Timestamp("2025-01-01")
    created_at = pd.date_range(base_ts, periods=n, freq="30s").astype(str)

    # --- Bettor-level aggregates ---
    bs_pnl = np.where(is_sharp_ticket, RNG.normal(-500, 200, n), RNG.normal(50, 100, n))
    bs_margin = np.where(
        is_sharp_ticket, RNG.normal(-0.05, 0.02, n), RNG.normal(0.02, 0.03, n)
    )
    bs_stake = np.abs(RNG.lognormal(5, 1, n))
    mean_stake_size = np.abs(RNG.lognormal(2, 0.8, n))

    df = pd.DataFrame(
        {
            # Metadata
            "ticket_id": [f"t_{i:06d}" for i in range(n)],
            "bettor_id": [bettor_ids[idx] for idx in bettor_idx],
            "created_at": created_at,
            "accepted_at": created_at,
            "rejected_at": np.where(rejected_mask, created_at, None),
            "home_team": "Team A",
            "away_team": "Team B",
            "tournament": "Test League",
            "client_name": "TestClient",
            "market_params": None,
            # Ticket features
            "stake": stake,
            "selection_odds": selection_odds,
            "cashout_stake": RNG.random(n) * stake * 0.5,
            "market_type_id": RNG.integers(1, 6, size=n),
            "market_name": RNG.choice(markets, size=n),
            "market_selection": RNG.choice(selections, size=n),
            "sport": sport_choices,
            "sport_id": [sport_id_map[s] for s in sport_choices],
            "tournament_id": RNG.integers(100, 500, size=n),
            "match_id": RNG.integers(1000, 5000, size=n),
            "client_id": RNG.choice(clients, size=n),
            "bos": RNG.normal(1.0, 0.1, size=n),
            "oaf": RNG.normal(0.0, 0.02, size=n),
            "ots_risk_tier_id": RNG.choice([1, 2, 3, 4, 5], size=n),
            "pending_delay": RNG.choice([0, 2, 5, 10], size=n),
            # Leaked / outcome columns (present in test set but excluded from features)
            "odds_after_10": odds_after_10,
            "odds_after_30": odds_after_10 + RNG.normal(0, 0.01, n),
            "odds_after_90": odds_after_10 + RNG.normal(0, 0.02, n),
            "ticket_state": ticket_state,
            "pnl": pnl,
            "reject_reason": reject_reason,
            # Bettor aggregates
            "bs_pnl": bs_pnl,
            "bs_margin": bs_margin,
            "bs_stake": bs_stake,
            "bs_avg_odds_after_10": RNG.normal(-0.005, 0.02, n),
            "bs_avg_odds_after_30": RNG.normal(-0.005, 0.025, n),
            "bs_avg_odds_after_90": RNG.normal(-0.005, 0.03, n),
            "bs_rejected_stake": np.abs(RNG.lognormal(3, 1, n)),
            "bs_rejected_pnl": RNG.normal(-50, 100, n),
            "total_rejected_stake": np.abs(RNG.lognormal(4, 1, n)),
            "total_rejected_pnl": RNG.normal(-100, 200, n),
            "n_reject_reasons": RNG.integers(0, 5, size=n),
            "avg_rejected_odds_after_10": RNG.normal(-0.01, 0.02, n),
            "avg_rejected_odds_after_30": RNG.normal(-0.01, 0.025, n),
            "avg_rejected_odds_after_90": RNG.normal(-0.01, 0.03, n),
            "dominant_risk_tier": RNG.choice([1, 2, 3, 4, 5], size=n),
            "risk_tier_total_pnl": RNG.normal(-200, 500, n),
            "risk_tier_total_volume": np.abs(RNG.lognormal(6, 1, n)),
            "risk_tier_avg_margin": RNG.normal(-0.02, 0.05, n),
            "mean_stake_size": mean_stake_size,
        }
    )

    return df


def main():
    print(f"Generating {N_ROWS:,} synthetic rows...")
    full_df = generate_synthetic_data(N_ROWS)

    # Temporal split: 70/15/15
    full_df = full_df.sort_values("created_at").reset_index(drop=True)
    n = len(full_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = full_df.iloc[:train_end]
    val_df = full_df.iloc[train_end:val_end]
    test_df = full_df.iloc[val_end:]

    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    # Prepare features
    train_with_target = add_target(train_df, TARGET_COL)
    X_train, y_train, cat_maps = prepare_features(
        train_with_target, fit_categories=True
    )

    val_with_target = add_target(val_df, TARGET_COL)
    X_val, y_val, _ = prepare_features(
        val_with_target, fit_categories=False, category_maps=cat_maps
    )

    feature_names = X_train.columns.tolist()
    cat_indices = get_categorical_indices(feature_names)
    print(f"  {len(feature_names)} features, {len(cat_indices)} categoricals")

    # Convert categorical columns to category dtype for LightGBM
    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
            X_val[col] = X_val[col].astype("category")

    # Train a small LightGBM
    print("Training LightGBM (tiny)...")
    dtrain = lgb.Dataset(
        X_train, label=y_train, categorical_feature=cat_indices, free_raw_data=False
    )
    dval = lgb.Dataset(
        X_val,
        label=y_val,
        categorical_feature=cat_indices,
        reference=dtrain,
        free_raw_data=False,
    )

    params = {
        "objective": "huber",
        "metric": "mae",
        "num_leaves": 15,
        "learning_rate": 0.1,
        "n_estimators": 50,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "verbose": -1,
    }

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=50,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.log_evaluation(period=25),
            lgb.early_stopping(stopping_rounds=10),
        ],
    )

    # Validation metrics
    val_preds = booster.predict(X_val)
    mae = float(np.mean(np.abs(val_preds - y_val)))
    rmse = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))
    print(f"  Val MAE: {mae:.5f}, RMSE: {rmse:.5f}")

    importance = dict(
        zip(feature_names, booster.feature_importance(importance_type="gain").tolist())
    )
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    metrics = {
        "val_mae": mae,
        "val_rmse": rmse,
        "best_iteration": booster.best_iteration,
        "n_features": len(feature_names),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(test_df),
        "feature_importance_top20": dict(list(importance.items())[:20]),
    }

    # Fit calibrator
    print("Fitting calibrator...")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_preds, y_val.values)

    cal_preds = iso.predict(val_preds)
    pre_mae = float(np.mean(np.abs(val_preds - y_val.values)))
    post_mae = float(np.mean(np.abs(cal_preds - y_val.values)))
    print(f"  Calibration MAE: {pre_mae:.5f} -> {post_mae:.5f}")

    # Save all artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving artifacts to {MODEL_DIR}...")

    booster.save_model(str(MODEL_DIR / "model.txt"))

    with open(MODEL_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(iso, f)

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(MODEL_DIR / "cat_maps.json", "w") as f:
        json.dump(cat_maps, f)

    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    test_df.to_parquet(MODEL_DIR / "test_set.parquet", index=False)
    print(f"  Saved test_set.parquet ({len(test_df):,} rows)")

    # Run OPE on the test set
    print("Running OPE...")
    test_with_target = add_target(test_df, TARGET_COL)
    X_test, y_test, _ = prepare_features(
        test_with_target, fit_categories=False, category_maps=cat_maps
    )
    for col in CATEGORICAL_FEATURES:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")
    raw_preds = booster.predict(X_test)
    cal_test_preds = iso.predict(raw_preds)

    test_metrics = regression_metrics(y_test.values, cal_test_preds)
    print(f"  Test MAE: {test_metrics['mae']:.5f}, RMSE: {test_metrics['rmse']:.5f}")

    pred_series = pd.Series(cal_test_preds, index=test_with_target.index)
    tiers = assign_tiers(
        pred_series,
        settings.thresholds.higher,
        settings.thresholds.static_lower,
        settings.thresholds.lower_skip,
    )
    print(f"  Tier distribution:\n{tiers.value_counts().to_string()}")

    sim_df = simulate_policy(test_with_target, tiers)
    summary = policy_summary(sim_df)
    print(f"  Skip rate: {summary['skip_rate']:.1%}")
    print(f"  PnL delta: {summary['pnl_delta']:,.0f}")

    # Save OPE results
    ope_dir = MODEL_DIR / "ope"
    ope_dir.mkdir(exist_ok=True)

    all_ope = {**test_metrics, **summary}
    with open(ope_dir / "ope_metrics.json", "w") as f:
        json.dump(all_ope, f, indent=2)

    sim_df.to_parquet(ope_dir / "simulation_results.parquet", index=False)

    # Higher review sample
    higher_tickets = sim_df[sim_df["model_tier"] == "HIGHER"].copy()
    higher_tickets["predicted_clv"] = cal_test_preds[tiers == "HIGHER"]
    if len(higher_tickets) > 100:
        worst = higher_tickets.nsmallest(50, "predicted_clv")
        rest = higher_tickets.drop(worst.index)
        sample = pd.concat([worst, rest.sample(min(50, len(rest)), random_state=42)])
    else:
        sample = higher_tickets
    sample.to_parquet(ope_dir / "higher_review_sample.parquet", index=False)
    print(f"  Saved {len(sample)} HIGHER review tickets")

    print(f"\nDone! Model artifacts at: {MODEL_DIR}")
    print("Run the app:  streamlit run app.py")


if __name__ == "__main__":
    main()
