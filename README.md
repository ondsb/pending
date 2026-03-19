# Pending Delay Model

LightGBM-based CLV prediction system for sports betting. Predicts `odds_after_10`
(closing line value) per ticket and assigns delay tiers (HIGHER / STATIC / LOWER / SKIP)
to manage risk from sharp bettors.

## Setup

Requires Python >= 3.11.

```bash
uv sync
```

## How to Use

### Quick start (no real data needed)

Generate a dummy model with 20k synthetic rows to test the dashboard:

```bash
python create_dummy_model.py
streamlit run app.py
```

### Full pipeline (with S3 access)

```bash
# 1. Ingest data from S3 (downloads CSV, converts to parquet, merges aggregates)
python convert_to_parquet.py

# For local dev, grab a small subset instead:
python convert_to_parquet.py --subset 50

# 2. (Optional) Inspect the dataset before training
python build_dataset.py --inspect

# 3. Train + calibrate + OPE in one command
python build_dataset.py --full

# Or run each step individually:
train --data data/tickets.parquet
calibrate
ope

# 4. Launch the dashboard
streamlit run app.py
```

### Validate an external model

For pre-trained models that were not trained by this pipeline:

```bash
python validate_model.py \
    --model /path/to/model.txt \
    --data data/tickets.parquet \
    --name my_model_v1
```

On machines with limited RAM (the full dataset is 150M rows / 24GB), use
`--max-row-groups` to read only the last N row groups instead of loading
everything into memory. Each row group is ~123k rows:

```bash
python validate_model.py \
    --model /path/to/model.txt \
    --data data/tickets.parquet \
    --name my_model_v1 \
    --max-row-groups 5     # ~615k rows, ~2-3 GB RAM
```

Other options:

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | inferred from path | Model directory name under `models/` |
| `--feature-names` | inferred from model | JSON file with feature column names |
| `--val-frac` | 0.3 | Fraction of data used for calibration |
| `--no-calibrate` | off | Skip isotonic calibration |
| `--max-row-groups` | all | Limit to last N parquet row groups |

### Pipeline scripts reference

| Command | Description |
|---------|-------------|
| `python convert_to_parquet.py` | S3 CSV to local parquet with aggregate merges |
| `python build_dataset.py --inspect` | Print dataset statistics (no training) |
| `python build_dataset.py --full` | Train + calibrate + OPE end-to-end |
| `train` | Train LightGBM (temporal split, memory-efficient) |
| `calibrate` | Fit isotonic calibrator on validation set |
| `ope` | Offline Policy Evaluation on test set |
| `python validate_model.py` | Validate an external pre-trained model |
| `python create_dummy_model.py` | Generate synthetic data + dummy model |
| `streamlit run app.py` | Launch the validation dashboard |

## Dashboard

```bash
streamlit run app.py
```

The dashboard auto-discovers models under `models/` (any directory with a `model.txt`).

**Tabs:**

- **Validation** -- Regression diagnostics: MAE, RMSE, correlation, predicted vs actual
  scatter, residual distribution, calibration analysis, feature importance.
- **Tier Simulator** -- Interactive threshold tuning with live PnL simulation.
  Adjust HIGHER/STATIC/LOWER/SKIP boundaries and see the impact in real time.
- **Segments** -- Per-segment performance drilldown by sport, market, risk tier,
  client, or odds bucket.
- **Review Queue** -- Browse HIGHER-tier tickets for manual inspection.
- **Data Explorer** -- DuckDB-backed data browser with filters and pagination.
- **Analytics** -- Dataset overview: ticket/bettor/sport counts, PnL breakdown,
  delay tier analysis, aggregate feature coverage.
- **Model Comparison** -- Side-by-side A/B comparison of two models (requires 2+
  models). Regression metric deltas, policy impact comparison, calibration overlay.

## Model Artifacts

After training or validation, each `models/<name>/` directory contains:

```
models/<name>/
  model.txt                      # LightGBM booster
  feature_names.json             # Ordered feature column names
  metrics.json                   # Validation metrics + feature importance
  calibrator.pkl                 # Isotonic calibrator
  cat_maps.json                  # Categorical feature mappings
  val_set.parquet                # Validation split
  test_set.parquet               # Test split
  ope/
    ope_metrics.json             # Regression + policy simulation metrics
    simulation_results.parquet   # Full simulation output
    higher_review_sample.parquet # HIGHER-tier tickets for review
    pred_vs_actual.png           # Scatter plot
    calibration.png              # Calibration plot
    feature_importance.png       # Feature importance chart
```

## Configuration

All settings are in `src/pending_delay/config.py` (pydantic-settings):

| Section | Key defaults |
|---------|-------------|
| Paths | `data/` for data, `models/` for artifacts |
| Model | Huber loss, 63 leaves, lr=0.05, 1000 rounds, early stop 50 |
| Thresholds | HIGHER: -0.02, STATIC/LOWER: -0.005, LOWER/SKIP: 0.005 |
| Split | 70/15/15 train/val/test, max 10M training rows |
| Filters | `bos >= 1.0`, `0 < pending_delay <= 9` |
