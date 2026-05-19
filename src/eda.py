"""Generate exploratory data analysis figures for the project report."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_pipeline import DEFAULT_DATA_PATH, ROOT, SPEND_COLS, load_raw_data


DEFAULT_OUT_DIR = ROOT / "reports" / "figures" / "eda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA figures for Spaceship Titanic.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def save_bar(series: pd.Series, title: str, ylabel: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    series.plot(kind="bar", ax=ax, color="#4477aa")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_target_distribution(train: pd.DataFrame, out_dir: Path) -> None:
    counts = train["Transported"].value_counts().sort_index()
    counts.index = ["False", "True"]
    save_bar(counts, "Target distribution", "Passengers", out_dir / "target_distribution.png")


def plot_missing_values(train: pd.DataFrame, test: pd.DataFrame, out_dir: Path) -> None:
    missing = pd.DataFrame({
        "train": train.isna().mean().mul(100),
        "test": test.isna().mean().mul(100),
    }).sort_values("train", ascending=False)
    missing = missing[missing.max(axis=1) > 0]

    fig, ax = plt.subplots(figsize=(9, 5))
    missing.plot(kind="bar", ax=ax, color=["#4477aa", "#cc6677"])
    ax.set_title("Missing value rate by column")
    ax.set_ylabel("Missing rate (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "missing_values.png", dpi=160)
    plt.close(fig)


def plot_numeric_distributions(train: pd.DataFrame, out_dir: Path) -> None:
    numeric_cols = ["Age", *SPEND_COLS]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for ax, col in zip(axes, numeric_cols):
        values = train[col].dropna()
        ax.hist(values, bins=35, color="#228833", alpha=0.8)
        ax.set_title(col)
        ax.set_ylabel("Count")
    fig.suptitle("Numeric feature distributions", y=0.99)
    fig.tight_layout()
    fig.savefig(out_dir / "numeric_distributions.png", dpi=160)
    plt.close(fig)


def plot_spend_by_target(train: pd.DataFrame, out_dir: Path) -> None:
    df = train.copy()
    df["TotalSpend"] = df[SPEND_COLS].fillna(0).sum(axis=1)
    grouped = df.groupby("Transported")[["TotalSpend", *SPEND_COLS]].median().T
    grouped.columns = ["False", "True"]

    fig, ax = plt.subplots(figsize=(9, 5))
    grouped.plot(kind="bar", ax=ax, color=["#cc6677", "#4477aa"])
    ax.set_title("Median spending by target")
    ax.set_ylabel("Median spend")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_dir / "spend_by_target.png", dpi=160)
    plt.close(fig)


def plot_categorical_target_rates(train: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col in zip(axes.ravel(), ["HomePlanet", "CryoSleep", "Destination", "VIP"]):
        rates = train.groupby(col, dropna=False)["Transported"].mean().sort_values(ascending=False)
        rates.plot(kind="bar", ax=ax, color="#aa3377")
        ax.set_title(f"Transported rate by {col}")
        ax.set_ylabel("Target rate")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "categorical_target_rates.png", dpi=160)
    plt.close(fig)


def plot_correlation_heatmap(train: pd.DataFrame, out_dir: Path) -> None:
    df = train.copy()
    df["TotalSpend"] = df[SPEND_COLS].fillna(0).sum(axis=1)
    corr_cols = ["Transported", "Age", "TotalSpend", *SPEND_COLS]
    corr = df[corr_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(corr.index)), corr.index)
    for row in range(len(corr.index)):
        for col in range(len(corr.columns)):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlation heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=160)
    plt.close(fig)


def write_summary(train: pd.DataFrame, test: pd.DataFrame, out_dir: Path) -> None:
    summary = [
        f"train_rows,{len(train)}",
        f"test_rows,{len(test)}",
        f"target_true_rate,{train['Transported'].mean():.6f}",
        f"train_missing_cells,{int(train.isna().sum().sum())}",
        f"test_missing_cells,{int(test.isna().sum().sum())}",
    ]
    (out_dir / "eda_summary.csv").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train, test = load_raw_data(args.data)

    plot_target_distribution(train, args.out_dir)
    plot_missing_values(train, test, args.out_dir)
    plot_numeric_distributions(train, args.out_dir)
    plot_spend_by_target(train, args.out_dir)
    plot_categorical_target_rates(train, args.out_dir)
    plot_correlation_heatmap(train, args.out_dir)
    write_summary(train, test, args.out_dir)

    print(f"EDA figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
