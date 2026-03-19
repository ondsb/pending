"""LightGBM training with temporal split via DuckDB.

Uses lgb.Sequence for batched dataset construction so training data never
fully materializes in memory.
"""

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

from pending_delay.config import settings
from pending_delay.features.engineering import encode_features_to_numpy
from pending_delay.features.filters import apply_filters
from pending_delay.schema import TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def split_to_parquet(
    data_path: Path,
    model_dir: Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[Path, Path, Path]:
    """Temporal split by row position — no sorting needed.

    The source parquet preserves Athena query order (by created_at).
    Streams through row groups and assigns them to train/val/test splits
    based on cumulative row count.  Peak memory = one row group.
    """
    train_path = model_dir / "train_set.parquet"
    val_path = model_dir / "val_set.parquet"
    test_path = model_dir / "test_set.parquet"

    pf = pq.ParquetFile(data_path)
    total = pf.metadata.num_rows
    train_end = int(total * train_frac)
    val_end = int(total * (train_frac + val_frac))

    log.info(f"Total rows: {total:,}")
    log.info(
        f"Split: train={train_end:,}, val={val_end - train_end:,}, test={total - val_end:,}"
    )

    import pyarrow as pa
    import pyarrow.compute as pc

    writers = {}
    paths = {"train": train_path, "val": val_path, "test": test_path}
    rows_seen = 0

    for i in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(i)

        # Filter nulls and infs in target
        target = table.column(TARGET_COL)
        mask = pc.and_(pc.is_valid(target), pc.is_finite(target))
        table = table.filter(mask)

        # Apply configurable pre-training filters (e.g. bos >= 1)
        table = apply_filters(table, settings.filter.rules)

        # Add engineered features
        stake = table.column("stake")
        mean_ss = table.column("mean_stake_size")
        stake_ratio = pc.if_else(
            pc.and_(pc.is_valid(mean_ss), pc.greater(mean_ss, 0)),
            pc.divide(stake, mean_ss),
            pa.scalar(None, type=pa.float64()),
        )
        table = table.append_column("stake_ratio", stake_ratio)

        odds = table.column("selection_odds")
        buckets = pc.if_else(
            pc.less(odds, 1.3),
            pa.scalar("heavy_fav"),
            pc.if_else(
                pc.less(odds, 1.7),
                pa.scalar("slight_fav"),
                pc.if_else(
                    pc.less(odds, 2.2),
                    pa.scalar("even"),
                    pc.if_else(
                        pc.less(odds, 3.5),
                        pa.scalar("underdog"),
                        pa.scalar("longshot"),
                    ),
                ),
            ),
        )
        table = table.append_column("odds_bucket", buckets)

        n = table.num_rows

        # Figure out which rows go to which split
        for start, end, split in [
            (0, max(0, min(n, train_end - rows_seen)), "train"),
            (
                max(0, min(n, train_end - rows_seen)),
                max(0, min(n, val_end - rows_seen)),
                "val",
            ),
            (max(0, min(n, val_end - rows_seen)), n, "test"),
        ]:
            if start >= end:
                continue
            chunk = table.slice(start, end - start)
            if split not in writers:
                writers[split] = pq.ParquetWriter(
                    str(paths[split]),
                    chunk.schema,
                    compression="zstd",
                )
            writers[split].write_table(chunk)

        rows_seen += n
        if i % 20 == 0:
            log.info(f"  processed {rows_seen:,} / {total:,} rows")

    for w in writers.values():
        w.close()

    log.info(f"Splits written to {model_dir}")
    return train_path, val_path, test_path


class ParquetBatchSequence(lgb.Sequence):
    """Lazy parquet reader for LightGBM — loads one row group at a time.

    Supports both row-level access (used during sampling) and batch-level
    access (used during construction via batch_sizes()).
    Caches the current row group so sequential access is fast.
    """

    def __init__(
        self, parquet_path: Path, feature_names: list[str], max_rows: int | None = None
    ):
        self.path = parquet_path
        self.feature_names = feature_names
        self.pf = pq.ParquetFile(parquet_path)
        self.n_row_groups = self.pf.metadata.num_row_groups
        self._batch_sizes = [
            self.pf.metadata.row_group(i).num_rows for i in range(self.n_row_groups)
        ]
        # Trim to max_rows by keeping only enough row groups
        if max_rows is not None:
            trimmed = []
            total = 0
            for size in self._batch_sizes:
                remaining = max_rows - total
                if remaining <= 0:
                    break
                trimmed.append(min(size, remaining))
                total += trimmed[-1]
            self._batch_sizes = trimmed
            self.n_row_groups = len(trimmed)
        self._offsets = np.cumsum([0] + self._batch_sizes)
        self._cached_rg_idx = -1
        self._cached_data = None

    def _load_row_group(self, rg_idx):
        if rg_idx != self._cached_rg_idx:
            table = self.pf.read_row_group(rg_idx, columns=self.feature_names)
            data = encode_features_to_numpy(table.to_pandas())
            # Trim the last row group if max_rows cut it short
            expected = self._batch_sizes[rg_idx]
            if data.shape[0] > expected:
                data = data[:expected]
            self._cached_data = data
            self._cached_rg_idx = rg_idx
        return self._cached_data

    def __len__(self):
        return int(self._offsets[-1])

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, _ = idx.indices(int(self._offsets[-1]))
            parts = []
            for rg_idx in range(self.n_row_groups):
                rg_start = int(self._offsets[rg_idx])
                rg_end = int(self._offsets[rg_idx + 1])
                if rg_end <= start:
                    continue
                if rg_start >= stop:
                    break
                batch = self._load_row_group(rg_idx)
                local_start = max(0, start - rg_start)
                local_end = min(rg_end - rg_start, stop - rg_start)
                parts.append(batch[local_start:local_end])
            return np.vstack(parts)
        else:
            rg_idx = int(np.searchsorted(self._offsets[1:], idx, side="right"))
            row_in_rg = idx - int(self._offsets[rg_idx])
            batch = self._load_row_group(rg_idx)
            return batch[row_in_rg]

    def batch_sizes(self):
        return self._batch_sizes


def _read_labels(parquet_path: Path, max_rows: int | None = None) -> np.ndarray:
    """Read only the target column from parquet — cheap columnar read."""
    y = pq.read_table(parquet_path, columns=[TARGET_COL]).to_pandas()[TARGET_COL].values
    # Replace any remaining inf with 0 (shouldn't happen after split filtering, but safety net)
    y = np.where(np.isfinite(y), y, 0.0)
    if max_rows is not None:
        y = y[:max_rows]
    return y


def _get_feature_names(parquet_path: Path) -> list[str]:
    """Determine feature columns from the split parquet schema."""
    pf = pq.ParquetFile(parquet_path)
    all_cols = pf.schema_arrow.names
    configured = settings.feature.features
    return [c for c in configured if c in all_cols]


def train_model(
    data_path: Path,
    model_dir: Path | None = None,
) -> tuple[lgb.Booster, dict]:
    """Train LightGBM regression model on merged parquet data.

    Uses DuckDB for sorting/splitting and lgb.Sequence for batched dataset
    construction so the full dataset never lives in memory at once.

    Returns:
        Tuple of (trained booster, metrics dict).
    """
    model_dir = model_dir or settings.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    # Reuse existing splits if available, otherwise create them
    train_path = model_dir / "train_set.parquet"
    val_path = model_dir / "val_set.parquet"
    test_path = model_dir / "test_set.parquet"

    if train_path.exists() and val_path.exists() and test_path.exists():
        log.info("Reusing existing splits in %s", model_dir)
    else:
        train_path, val_path, test_path = split_to_parquet(
            data_path,
            model_dir,
            train_frac=settings.split.train_frac,
            val_frac=settings.split.val_frac,
        )

    # Determine feature names from parquet schema
    feature_names = _get_feature_names(train_path)

    log.info(f"Features: {len(feature_names)}")
    log.info(f"Feature list: {feature_names}")

    # Read labels only (just one column — small)
    max_train = settings.split.max_train_rows
    y_train = _read_labels(train_path, max_rows=max_train)
    y_val = _read_labels(val_path)

    log.info(
        f"Target — train: {len(y_train):,} rows"
        + (f" (capped from max_train_rows={max_train:,})" if max_train else "")
    )
    log.info(f"Target — train: mean={y_train.mean():.4f} std={y_train.std():.4f}")
    log.info(f"Target — val:   mean={y_val.mean():.4f} std={y_val.std():.4f}")

    # Build LightGBM datasets via Sequence — one row group in memory at a time
    log.info("Building LightGBM train dataset (batched)...")
    train_seq = ParquetBatchSequence(train_path, feature_names, max_rows=max_train)
    dtrain = lgb.Dataset(
        train_seq,
        label=y_train,
        free_raw_data=True,
    )

    log.info("Building LightGBM val dataset (batched)...")
    val_seq = ParquetBatchSequence(val_path, feature_names)
    dval = lgb.Dataset(
        val_seq,
        label=y_val,
        reference=dtrain,
        free_raw_data=True,
    )

    mcfg = settings.model
    params = {
        "objective": mcfg.objective,
        "metric": mcfg.metric,
        "num_leaves": mcfg.num_leaves,
        "learning_rate": mcfg.learning_rate,
        "feature_fraction": mcfg.feature_fraction,
        "bagging_fraction": mcfg.bagging_fraction,
        "bagging_freq": mcfg.bagging_freq,
        "min_child_samples": mcfg.min_child_samples,
        "verbose": mcfg.verbose,
    }

    log.info("Training LightGBM...")
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=mcfg.n_estimators,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=mcfg.early_stopping_rounds),
        ],
    )

    # Evaluate on val set — read in batches to avoid memory spike
    log.info("Evaluating on val set...")
    val_preds_parts = []
    val_pf = pq.ParquetFile(val_path)
    for i in range(val_pf.metadata.num_row_groups):
        batch = encode_features_to_numpy(
            val_pf.read_row_group(i, columns=feature_names).to_pandas()
        )
        val_preds_parts.append(booster.predict(batch))
        del batch
    val_preds = np.concatenate(val_preds_parts)
    del val_preds_parts

    mae = float(np.mean(np.abs(val_preds - y_val)))
    rmse = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))

    # Feature importance
    importance = dict(
        zip(feature_names, booster.feature_importance(importance_type="gain").tolist())
    )
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    metrics = {
        "val_mae": mae,
        "val_rmse": rmse,
        "best_iteration": booster.best_iteration,
        "n_features": len(feature_names),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "features_used": feature_names,
        "feature_importance_top20": dict(list(importance.items())[:20]),
    }

    log.info(f"Val MAE: {mae:.5f}, Val RMSE: {rmse:.5f}")
    log.info(f"Best iteration: {booster.best_iteration}")
    log.info(f"Top 5 features: {list(importance.keys())[:5]}")

    # Save artifacts
    booster.save_model(str(model_dir / "model.txt"))
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    log.info(f"Saved model artifacts to {model_dir}")
    return booster, metrics


def main():
    parser = argparse.ArgumentParser(description="Train pending delay model")
    parser.add_argument(
        "--data", type=Path, default=settings.data_dir / "tickets.parquet"
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()
    train_model(args.data, args.model_dir)


if __name__ == "__main__":
    main()
