"""Centralized configuration for the pending delay model."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class S3Config(BaseSettings):
    src_bucket: str = "oddin-statistics-data"
    dst_bucket: str = "oddin-training-artifacts"
    csv_key: str = "athena_results/81833cb9-ecaf-4e24-873b-bad30a69a898.csv"
    parquet_key: str = "data/pending/tickets.parquet"
    region: str = "eu-west-1"
    src_profile: str = "oddin"
    dst_profile: str = "oddin-model-training"

    aggregates: dict[str, str] = {
        "bettor_stats": "athena_results/ba2e1719-6078-4dfb-8f87-7fc93b530951.csv",
        "rejected": "athena_results/afb9b31b-8127-48c0-8454-7e6712f3bffb.csv",
        "risk_tier": "athena_results/0d396ecc-4719-472e-a5bd-2ec0f1b8e4c7.csv",
        "stake_size": "athena_results/adba3175-c0be-4ed7-a30d-0b6753f59b1c.csv",
    }


class ModelConfig(BaseSettings):
    objective: str = "huber"
    metric: str = "mae"
    num_leaves: int = 63
    learning_rate: float = 0.05
    n_estimators: int = 1000
    early_stopping_rounds: int = 50
    min_child_samples: int = 100
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    verbose: int = -1


class ThresholdConfig(BaseSettings):
    """Delay tier thresholds applied to predicted CLV (odds_after_10).

    Negative predicted CLV = line moved in bettor's favor = sharp/toxic.
    """

    higher: float = -0.02  # pred < higher → HIGHER delay (punitive)
    static_lower: float = -0.005  # higher <= pred < static_lower → STATIC
    lower_skip: float = 0.005  # static_lower <= pred < lower_skip → LOWER delay
    # pred >= lower_skip → SKIP (recreational, bypass)


class SplitConfig(BaseSettings):
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac = 1 - train_frac - val_frac = 0.15


class Settings(BaseSettings):
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = Field(default=None)
    model_dir: Path = Field(default=None)

    s3: S3Config = S3Config()
    model: ModelConfig = ModelConfig()
    thresholds: ThresholdConfig = ThresholdConfig()
    split: SplitConfig = SplitConfig()

    def model_post_init(self, __context):
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.model_dir is None:
            self.model_dir = self.project_root / "models"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
