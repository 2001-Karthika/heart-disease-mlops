"""Exploratory data analysis: produces figures and a short text summary in reports/."""
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config  # noqa: E402
from src.data import load_clean  # noqa: E402

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {0: "#4C78A8", 1: "#E45756"}


def _save(fig, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_class_balance(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[config.TARGET].value_counts().sort_index()
    bars = ax.bar(["No disease (0)", "Disease (1)"], counts.values,
                  color=[PALETTE[0], PALETTE[1]])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                f"{val} ({val / len(df):.1%})", ha="center")
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    return _save(fig, "eda_class_balance.png", out_dir)


def plot_histograms(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(data=df, x=col, hue=config.TARGET, bins=20, kde=(col != "ca"),
                     palette=PALETTE, ax=ax, element="step")
        ax.set_title(f"Distribution of {col}")
    fig.suptitle("Numeric feature distributions by target", y=1.02)
    return _save(fig, "eda_histograms.png", out_dir)


def plot_categorical(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = config.CATEGORICAL_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), cols):
        sns.countplot(data=df, x=col, hue=config.TARGET, palette=PALETTE, ax=ax)
        ax.set_title(f"{col} vs target")
        ax.legend(title="target", loc="upper right")
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    return _save(fig, "eda_categorical.png", out_dir)


def plot_correlation(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation heatmap")
    return _save(fig, "eda_correlation_heatmap.png", out_dir)


def plot_boxplots(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), cols):
        sns.boxplot(data=df, x=config.TARGET, y=col, hue=config.TARGET,
                    palette=PALETTE, ax=ax, legend=False)
        ax.set_title(f"{col} by target")
    return _save(fig, "eda_boxplots.png", out_dir)


def plot_missing(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if miss.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.axis("off")
    else:
        miss.plot(kind="bar", ax=ax, color="#F58518")
        ax.set_ylabel("Missing values")
        ax.set_title("Missing values per column")
    return _save(fig, "eda_missing_values.png", out_dir)


def write_summary(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = df[config.TARGET].value_counts().sort_index()
    corr_t = (df.corr(numeric_only=True)[config.TARGET]
              .drop(config.TARGET).sort_values(key=abs, ascending=False))
    lines = [
        "# EDA summary",
        "",
        f"- Rows: {len(df)}, columns: {df.shape[1]}",
        f"- Class counts: no disease={counts.get(0, 0)}, disease={counts.get(1, 0)} "
        f"({counts.get(1, 0) / len(df):.1%} positive)",
        f"- Missing values per column: {df.isna().sum()[df.isna().sum() > 0].to_dict()}",
        "",
        "## Descriptive statistics",
        "",
        df.describe().T.round(2).to_markdown()
        if _has_tabulate() else df.describe().T.round(2).to_string(),
        "",
        "## Correlation with target (absolute order)",
        "",
        corr_t.round(3).to_string(),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def run_eda(df: Optional[pd.DataFrame] = None, out_dir: Optional[Path] = None) -> list:
    df = load_clean() if df is None else df
    out_dir = Path(out_dir or config.FIG_DIR)
    paths = [
        plot_class_balance(df, out_dir),
        plot_histograms(df, out_dir),
        plot_categorical(df, out_dir),
        plot_correlation(df, out_dir),
        plot_boxplots(df, out_dir),
        plot_missing(df, out_dir),
        write_summary(df, config.REPORTS_DIR / "eda_summary.md"),
    ]
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for p in run_eda():
        print(p)
