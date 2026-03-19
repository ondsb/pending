"""Regression metrics and evaluation plots."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression evaluation metrics."""
    errors = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "median_ae": float(np.median(np.abs(errors))),
        "mean_pred": float(np.mean(y_pred)),
        "mean_actual": float(np.mean(y_true)),
        "std_pred": float(np.std(y_pred)),
        "std_actual": float(np.std(y_true)),
        "correlation": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0,
    }


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Predicted vs Actual CLV",
) -> None:
    """Scatter plot of predicted vs actual CLV values."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Subsample for readability if too many points
    n = len(y_true)
    if n > 10_000:
        idx = np.random.default_rng(42).choice(n, 10_000, replace=False)
        y_true_plot, y_pred_plot = y_true[idx], y_pred[idx]
    else:
        y_true_plot, y_pred_plot = y_true, y_pred

    ax.scatter(y_true_plot, y_pred_plot, alpha=0.1, s=3)

    # Perfect calibration line
    lims = [
        min(y_true_plot.min(), y_pred_plot.min()),
        max(y_true_plot.max(), y_pred_plot.max()),
    ]
    ax.plot(lims, lims, "r--", alpha=0.8, label="Perfect")
    ax.set_xlabel("Actual CLV (odds_after_10)")
    ax.set_ylabel("Predicted CLV")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved scatter plot to {output_path}")


def plot_calibration_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    n_bins: int = 20,
    title: str = "Binned Calibration",
) -> pd.DataFrame:
    """Binned calibration plot: mean predicted vs mean actual per decile."""
    df = pd.DataFrame({"pred": y_pred, "actual": y_true})
    df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")

    binned = (
        df.groupby("bin", observed=True)
        .agg(
            n=("actual", "count"),
            mean_pred=("pred", "mean"),
            mean_actual=("actual", "mean"),
            std_actual=("actual", "std"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(binned))
    ax.bar(x, binned["mean_actual"], alpha=0.5, label="Actual mean", width=0.4, align="edge")
    ax.bar([i + 0.4 for i in x], binned["mean_pred"], alpha=0.5, label="Predicted mean", width=0.4, align="edge")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.mean_pred:.3f}" for r in binned.itertuples()], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Predicted CLV (bin center)")
    ax.set_ylabel("Mean CLV")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved calibration plot to {output_path}")

    return binned


def plot_feature_importance(
    importance: dict[str, float],
    output_path: Path,
    top_n: int = 20,
    title: str = "Feature Importance (Gain)",
) -> None:
    """Horizontal bar chart of top feature importances."""
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, values = zip(*reversed(sorted_imp))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(names)), values)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Importance (Gain)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved feature importance plot to {output_path}")
