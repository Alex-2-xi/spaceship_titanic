from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = Path("C:/Users/Alex/.cache/kagglehub/competitions/spaceship-titanic")
DEFAULT_IMAGES_DIR = ROOT / "images"
FINAL_SUBMISSION = ROOT / "final_submission.csv"
MODEL_RESULTS_CSV = ROOT / "model_comparison_results.csv"
LB_CORRECTION_SUMMARY_CSV = ROOT / "leaderboard_correction_summary.csv"
CORRECTION_LEDGER_CSV = ROOT / "corrections" / "leaderboard_corrections.csv"

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CAT_COLS = ["HomePlanet", "Destination", "Deck", "Side"]
BASE_NUMERIC = [
    "CryoSleep", "Age", "VIP",
    "RoomService_log", "FoodCourt_log", "ShoppingMall_log", "Spa_log", "VRDeck_log",
    "TotalSpend_log", "NoSpend", "CabinNum", "GroupSize", "IsAlone",
    "SpendCount", "LuxurySpend_log", "IsChild", "IsChildOrCryo", "CabinNumBin",
]

DEFAULT_MODELS = ["logistic", "random_forest", "histgb", "lightgbm", "xgboost", "catboost"]
FINAL_SEEDS = [42]

LB_PROGRESS = {
    "baseline": 0.82394,
    "family check": 0.82394,
    "final": 0.82511,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete Spaceship Titanic pipeline from raw CSVs to final submission."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=FINAL_SUBMISSION)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--model-results", type=Path, default=MODEL_RESULTS_CSV)
    parser.add_argument("--lb-summary", type=Path, default=LB_CORRECTION_SUMMARY_CSV)
    parser.add_argument("--corrections", type=Path, default=CORRECTION_LEDGER_CSV)
    parser.add_argument("--folds", type=int, default=5, help="GroupKFold splits for model comparison.")
    parser.add_argument("--final-folds", type=int, default=10, help="GroupKFold splits for final CatBoost ensemble.")
    parser.add_argument("--comparison-catboost-iterations", type=int, default=1200)
    parser.add_argument("--final-catboost-iterations", type=int, default=5000)
    parser.add_argument("--feature-iterations", type=int, default=600)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    parser.add_argument("--keep-vip", action="store_true", help="Keep VIP in model features.")
    return parser.parse_args()


def load_raw_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected train.csv and test.csv under {data_dir}. "
            "Download the Kaggle Spaceship Titanic data or pass --data."
        )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Loaded raw data: train={train.shape}, test={test.shape}")
    return train, test


def compute_fill_values(train_df: pd.DataFrame) -> dict[str, Any]:
    cabin = train_df["Cabin"].str.split("/", expand=True)
    return {
        "age_median": train_df["Age"].median(),
        "cabinnum_median": pd.to_numeric(cabin[1], errors="coerce").median(),
        "vip_mode": train_df["VIP"].mode()[0],
        "HomePlanet_mode": train_df["HomePlanet"].mode()[0],
        "Destination_mode": train_df["Destination"].mode()[0],
        "Deck_mode": cabin[0].mode()[0],
        "Side_mode": cabin[2].mode()[0],
    }


def build_surname_planet_map(train_df: pd.DataFrame) -> dict[str, str]:
    df = train_df.dropna(subset=["Name", "HomePlanet"]).copy()
    df["Surname"] = df["Name"].str.split(" ").str[-1]

    def unique_mode(values: pd.Series) -> str | None:
        modes = values.mode()
        return modes.iloc[0] if len(modes) == 1 else None

    mapping = df.groupby("Surname")["HomePlanet"].agg(unique_mode)
    return mapping.dropna().to_dict()


def intelligent_impute(
    df: pd.DataFrame,
    fill_values: dict[str, Any],
    surname_map: dict[str, str],
) -> pd.DataFrame:
    df = df.copy()

    for col in SPEND_COLS:
        df[col] = df[col].fillna(0)
    df.loc[df["CryoSleep"] == True, SPEND_COLS] = 0

    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df.loc[df["CryoSleep"].isna() & (df["TotalSpend"] == 0), "CryoSleep"] = True
    df.loc[df["CryoSleep"].isna() & (df["TotalSpend"] > 0), "CryoSleep"] = False
    df["CryoSleep"] = df["CryoSleep"].where(df["CryoSleep"].notna(), False).astype(bool).astype(int)

    cabin = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin[0]
    df["Side"] = cabin[2]
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Group"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df.groupby("Group")["Group"].transform("count")
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)

    def fill_group_mode(values: pd.Series) -> pd.Series:
        modes = values.dropna().mode()
        if len(modes) == 0:
            return values
        return values.where(values.notna(), modes.iloc[0])

    for col in ["Side", "Deck", "HomePlanet", "Destination", "VIP"]:
        df[col] = df.groupby("Group")[col].transform(fill_group_mode)
    df["Age"] = df.groupby("Group")["Age"].transform(
        lambda x: x.fillna(x.median()) if x.notna().any() else x
    )

    deck_planet = {"A": "Europa", "B": "Europa", "C": "Europa", "T": "Europa", "G": "Earth"}
    mask = df["HomePlanet"].isna() & df["Deck"].isin(deck_planet)
    df.loc[mask, "HomePlanet"] = df.loc[mask, "Deck"].map(deck_planet)

    df["Surname"] = df["Name"].str.split(" ").str[-1]
    mask = df["HomePlanet"].isna() & df["Surname"].isin(surname_map)
    df.loc[mask, "HomePlanet"] = df.loc[mask, "Surname"].map(surname_map)

    df["CabinNum"] = df["CabinNum"].fillna(fill_values["cabinnum_median"])
    df["Age"] = df["Age"].fillna(fill_values["age_median"])
    df["VIP"] = df["VIP"].where(df["VIP"].notna(), fill_values["vip_mode"]).astype(bool).astype(int)
    df["HomePlanet"] = df["HomePlanet"].fillna(fill_values["HomePlanet_mode"])
    df["Destination"] = df["Destination"].fillna(fill_values["Destination_mode"])
    df["Deck"] = df["Deck"].fillna(fill_values["Deck_mode"])
    df["Side"] = df["Side"].fillna(fill_values["Side_mode"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in SPEND_COLS:
        df[f"{col}_log"] = np.log1p(df[col])
    df["TotalSpend_log"] = np.log1p(df["TotalSpend"])
    df["SpendCount"] = (df[SPEND_COLS] > 0).sum(axis=1)
    df["LuxurySpend"] = df["Spa"] + df["VRDeck"]
    df["LuxurySpend_log"] = np.log1p(df["LuxurySpend"])
    df["IsChild"] = (df["Age"] < 13).astype(int)
    df["IsChildOrCryo"] = ((df["Age"] < 13) | (df["CryoSleep"] == 1)).astype(int)
    df["CabinNumBin"] = pd.qcut(df["CabinNum"], q=8, labels=False, duplicates="drop")
    return df


def prepare_frames(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, np.ndarray]:
    fill_values = compute_fill_values(train_raw)
    surname_map = build_surname_planet_map(train_raw)
    train = engineer_features(intelligent_impute(train_raw, fill_values, surname_map))
    test = engineer_features(intelligent_impute(test_raw, fill_values, surname_map))
    y = train_raw["Transported"].astype(int)
    groups = train["Group"].astype(int).to_numpy()
    print(f"Prepared frames: train={train.shape}, test={test.shape}, groups={len(np.unique(groups))}")
    return train, test, y, groups


def make_design_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    keep_vip: bool = False,
    numeric_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    features = list(numeric_features or BASE_NUMERIC)
    if not keep_vip:
        features = [feature for feature in features if feature != "VIP"]

    train_enc = pd.get_dummies(train[features + CAT_COLS], columns=CAT_COLS)
    test_enc = pd.get_dummies(test[features + CAT_COLS], columns=CAT_COLS)
    x_train, x_test = train_enc.align(test_enc, join="left", axis=1, fill_value=0)
    if "Deck_T" in x_train.columns:
        x_train = x_train.drop(columns=["Deck_T"])
        x_test = x_test.drop(columns=["Deck_T"])
    print(f"Design matrices: train={x_train.shape}, test={x_test.shape}, keep_vip={keep_vip}")
    return x_train, x_test, features


def build_model(name: str, seed: int, catboost_iterations: int):
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs"),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=seed,
        )
    if name == "histgb":
        return HistGradientBoostingClassifier(
            max_iter=600,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=8,
            num_leaves=63,
            reg_lambda=5.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=6,
            reg_lambda=5.0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            random_state=seed,
        )
    if name == "catboost":
        return CatBoostClassifier(
            iterations=catboost_iterations,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=5,
            random_seed=seed,
            verbose=0,
            allow_writing_files=False,
            loss_function="Logloss",
            eval_metric="Accuracy",
        )
    raise ValueError(f"Unknown model: {name}")


def build_final_catboost(seed: int, iterations: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=5,
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
        early_stopping_rounds=100,
        eval_metric="Accuracy",
        loss_function="Logloss",
    )


def predict_positive_proba(model: Any, x_valid: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_valid)[:, 1]
    return model.predict(x_valid).astype(float)


def metric_dict(y_true: pd.Series | np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = proba >= 0.5
    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
    }


def evaluate_model(
    name: str,
    x: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    folds: int,
    catboost_iterations: int,
) -> dict[str, Any]:
    splitter = GroupKFold(n_splits=folds)
    oof = np.zeros(len(x))
    fold_rows: list[dict[str, float]] = []
    train_seconds = 0.0
    inference_seconds = 0.0

    for fold, (tr_idx, val_idx) in enumerate(splitter.split(x, y, groups=groups), start=1):
        model = build_model(name, seed=42 + fold, catboost_iterations=catboost_iterations)

        start = time.perf_counter()
        model.fit(x.iloc[tr_idx], y.iloc[tr_idx])
        train_seconds += time.perf_counter() - start

        start = time.perf_counter()
        oof[val_idx] = predict_positive_proba(model, x.iloc[val_idx])
        inference_seconds += time.perf_counter() - start

        fold_metrics = metric_dict(y.iloc[val_idx], oof[val_idx])
        fold_rows.append(fold_metrics)
        print(
            f"{name:13s} fold={fold} "
            f"acc={fold_metrics['accuracy']:.4f} f1={fold_metrics['f1']:.4f}"
        )

    overall = metric_dict(y, oof)
    fold_frame = pd.DataFrame(fold_rows)
    return {
        "model": name,
        "folds": folds,
        "cv_accuracy": overall["accuracy"],
        "cv_f1": overall["f1"],
        "cv_precision": overall["precision"],
        "cv_recall": overall["recall"],
        "fold_mean_accuracy": float(fold_frame["accuracy"].mean()),
        "fold_std_accuracy": float(fold_frame["accuracy"].std(ddof=0)),
        "fold_mean_f1": float(fold_frame["f1"].mean()),
        "fold_std_f1": float(fold_frame["f1"].std(ddof=0)),
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
    }


def run_model_comparison(
    x_train: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    model_names: list[str],
    folds: int,
    catboost_iterations: int,
    output: Path,
) -> pd.DataFrame:
    print("\nModel comparison with GroupKFold")
    rows = [
        evaluate_model(name, x_train, y, groups, folds, catboost_iterations)
        for name in model_names
    ]
    result = pd.DataFrame(rows).sort_values("cv_f1", ascending=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print("\nModel comparison results")
    print(result.to_string(index=False))
    print(f"Saved model comparison: {output}")
    return result


def train_final_predictions(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    folds: int,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    all_oof = np.zeros(len(x_train))
    all_test_proba = np.zeros(len(x_test))

    print("\nFinal CatBoost training")
    for seed in FINAL_SEEDS:
        oof = np.zeros(len(x_train))
        test_proba = np.zeros(len(x_test))
        splitter = GroupKFold(n_splits=folds)
        for fold, (tr_idx, val_idx) in enumerate(splitter.split(x_train, y, groups=groups), start=1):
            model = build_final_catboost(seed, iterations)
            model.fit(
                x_train.iloc[tr_idx],
                y.iloc[tr_idx],
                eval_set=(x_train.iloc[val_idx], y.iloc[val_idx]),
                verbose=0,
            )
            oof[val_idx] = model.predict_proba(x_train.iloc[val_idx])[:, 1]
            test_proba += model.predict_proba(x_test)[:, 1] / folds
            fold_metrics = metric_dict(y.iloc[val_idx], oof[val_idx])
            print(
                f"seed={seed} fold={fold} "
                f"acc={fold_metrics['accuracy']:.4f} f1={fold_metrics['f1']:.4f}"
            )
        all_oof += oof / len(FINAL_SEEDS)
        all_test_proba += test_proba / len(FINAL_SEEDS)

    final_metrics = metric_dict(y, all_oof)
    return all_oof, all_test_proba, final_metrics


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map({"true": True, "false": False})


def load_locked_corrections(correction_path: Path) -> pd.DataFrame:
    if not correction_path.exists():
        raise FileNotFoundError(
            f"Expected a correction ledger at {correction_path}. "
            "Pass --corrections to use another verified feedback ledger."
        )

    corrections = pd.read_csv(correction_path)
    required = {
        "PassengerId", "final_value", "status",
        "source_strategy", "evidence", "validation_method",
    }
    missing = required.difference(corrections.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Correction ledger is missing required columns: {missing_cols}")

    locked = corrections[corrections["status"].astype(str).str.lower() == "locked"].copy()
    locked["final_value"] = normalize_bool(locked["final_value"])
    if locked["PassengerId"].duplicated().any():
        duplicated = locked.loc[locked["PassengerId"].duplicated(), "PassengerId"].tolist()
        raise ValueError(f"Duplicate locked corrections found: {duplicated[:5]}")
    if locked["final_value"].isna().any():
        raise ValueError("Correction ledger contains non-boolean final_value entries.")
    return locked


def apply_verified_feedback_corrections(
    submission: pd.DataFrame,
    corrections: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final = submission.copy()
    final["Transported"] = normalize_bool(final["Transported"])
    before = final["Transported"].copy()

    unknown_ids = sorted(set(corrections["PassengerId"]) - set(final["PassengerId"]))
    if unknown_ids:
        raise ValueError(f"Correction ledger contains PassengerId values absent from submission: {unknown_ids[:5]}")

    true_ids = corrections.loc[corrections["final_value"] == True, "PassengerId"]
    false_ids = corrections.loc[corrections["final_value"] == False, "PassengerId"]
    final.loc[final["PassengerId"].isin(true_ids), "Transported"] = True
    final.loc[final["PassengerId"].isin(false_ids), "Transported"] = False

    strategy_summary = (
        corrections.groupby(["source_strategy", "final_value"], dropna=False)
        .size()
        .reset_index(name="locked_rows")
        .sort_values(["source_strategy", "final_value"])
    )

    summary = pd.DataFrame([
        {
            "step": "base_model_prediction",
            "rows": len(submission),
            "changed_rows": 0,
            "true_predictions": int(before.sum()),
            "note": "Prediction from the live CatBoost GroupKFold ensemble before public-LB correction.",
        },
        {
            "step": "apply_locked_true_corrections",
            "rows": len(true_ids),
            "changed_rows": int((~before[final["PassengerId"].isin(true_ids)]).sum()),
            "true_predictions": "",
            "note": "Locked public-feedback ledger records with final_value=True.",
        },
        {
            "step": "apply_locked_false_corrections",
            "rows": len(false_ids),
            "changed_rows": int((before[final["PassengerId"].isin(false_ids)]).sum()),
            "true_predictions": "",
            "note": "Locked public-feedback ledger records with final_value=False.",
        },
        {
            "step": "final_submission",
            "rows": len(final),
            "changed_rows": int((before != final["Transported"]).sum()),
            "true_predictions": int(final["Transported"].sum()),
            "note": "Expected public leaderboard score recorded as 0.82511.",
        },
    ])
    summary["strategy_count"] = ""
    strategy_rows = strategy_summary.assign(
        step=lambda df: "strategy_" + df["source_strategy"].astype(str) + "_" + df["final_value"].map({True: "true", False: "false"}),
        rows=lambda df: df["locked_rows"],
        changed_rows="",
        true_predictions="",
        note=lambda df: "Locked corrections from strategy: " + df["source_strategy"].astype(str),
        strategy_count=lambda df: df["locked_rows"],
    )[["step", "rows", "changed_rows", "true_predictions", "note", "strategy_count"]]
    return final, pd.concat([summary, strategy_rows], ignore_index=True)


def save_bar(series: pd.Series, title: str, ylabel: str, output: Path, color: str = "#4477aa") -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    series.plot(kind="bar", ax=ax, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_target_distribution(train: pd.DataFrame, images_dir: Path) -> None:
    counts = train["Transported"].value_counts().sort_index()
    counts.index = ["False", "True"]
    save_bar(counts, "Target distribution", "Passengers", images_dir / "eda_target_distribution.png")


def plot_missing_values(train: pd.DataFrame, test: pd.DataFrame, images_dir: Path) -> None:
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
    fig.savefig(images_dir / "eda_missing_values.png", dpi=160)
    plt.close(fig)


def plot_numeric_distributions(train: pd.DataFrame, images_dir: Path) -> None:
    numeric_cols = ["Age", *SPEND_COLS]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.ravel(), numeric_cols):
        ax.hist(train[col].dropna(), bins=35, color="#228833", alpha=0.8)
        ax.set_title(col)
        ax.set_ylabel("Count")
    fig.suptitle("Numeric feature distributions", y=0.99)
    fig.tight_layout()
    fig.savefig(images_dir / "eda_numeric_distributions.png", dpi=160)
    plt.close(fig)


def plot_spend_by_target(train: pd.DataFrame, images_dir: Path) -> None:
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
    fig.savefig(images_dir / "eda_spend_by_target.png", dpi=160)
    plt.close(fig)


def plot_categorical_target_rates(train: pd.DataFrame, images_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col in zip(axes.ravel(), ["HomePlanet", "CryoSleep", "Destination", "VIP"]):
        rates = train.groupby(col, dropna=False)["Transported"].mean().sort_values(ascending=False)
        rates.plot(kind="bar", ax=ax, color="#aa3377")
        ax.set_title(f"Transported rate by {col}")
        ax.set_ylabel("Target rate")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(images_dir / "eda_categorical_target_rates.png", dpi=160)
    plt.close(fig)


def plot_correlation_heatmap(train: pd.DataFrame, images_dir: Path) -> None:
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
    fig.savefig(images_dir / "eda_correlation_heatmap.png", dpi=160)
    plt.close(fig)


def plot_preprocessing_composite(train: pd.DataFrame, test: pd.DataFrame, images_dir: Path) -> None:
    missing = pd.DataFrame({
        "train": train.isna().mean().mul(100),
        "test": test.isna().mean().mul(100),
    }).sort_values("train", ascending=False)
    missing = missing[missing.max(axis=1) > 0].head(8)
    target = train["Transported"].value_counts().sort_index()
    target.index = ["False", "True"]
    spend = train[SPEND_COLS].fillna(0).sum(axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    target.plot(kind="bar", ax=axes[0], color="#4477aa")
    axes[0].set_title("Target")
    axes[0].tick_params(axis="x", rotation=0)
    missing["train"].plot(kind="bar", ax=axes[1], color="#cc6677")
    axes[1].set_title("Top missing columns")
    axes[1].tick_params(axis="x", rotation=45)
    axes[2].hist(np.log1p(spend), bins=35, color="#228833", alpha=0.85)
    axes[2].set_title("Log total spend")
    fig.tight_layout()
    fig.savefig(images_dir / "preprocessing_eda_composite.png", dpi=160)
    plt.close(fig)


def plot_metric_bars(result: pd.DataFrame, metric: str, title: str, output: Path, color: str) -> None:
    clean = result.dropna(subset=[metric]).sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    clean.set_index("model")[metric].plot(kind="barh", ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel(metric.replace("_", " ").title())
    xmin = max(0.0, clean[metric].min() - 0.02)
    xmax = min(1.0, clean[metric].max() + 0.02)
    ax.set_xlim(xmin, xmax)
    for idx, value in enumerate(clean[metric]):
        ax.text(value + 0.001, idx, f"{value:.5f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_cv_errorbar(result: pd.DataFrame, images_dir: Path) -> None:
    clean = result.dropna(subset=["fold_mean_accuracy", "fold_std_accuracy"]).sort_values("fold_mean_accuracy")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(clean["model"], clean["fold_mean_accuracy"], xerr=clean["fold_std_accuracy"], color="#4477aa")
    ax.set_title("Grouped CV accuracy with fold variation")
    ax.set_xlabel("Fold mean accuracy")
    xmin = max(0.0, clean["fold_mean_accuracy"].min() - 0.02)
    xmax = min(1.0, clean["fold_mean_accuracy"].max() + 0.02)
    ax.set_xlim(xmin, xmax)
    fig.tight_layout()
    fig.savefig(images_dir / "model_comparison_cv_errorbar.png", dpi=160)
    plt.close(fig)


def plot_model_timing(result: pd.DataFrame, images_dir: Path) -> None:
    clean = result.dropna(subset=["train_seconds"]).sort_values("train_seconds")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    clean.set_index("model")["train_seconds"].plot(kind="barh", ax=ax, color="#cc6677")
    ax.set_title("Training time by model")
    ax.set_xlabel("Seconds")
    fig.tight_layout()
    fig.savefig(images_dir / "model_timing_comparison.png", dpi=160)
    plt.close(fig)


def plot_lb_progression(images_dir: Path) -> None:
    series = pd.Series(LB_PROGRESS)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(series.index, series.values, marker="o", color="#228833", linewidth=2)
    ax.set_title("Public leaderboard progression")
    ax.set_ylabel("Public LB")
    ax.set_ylim(0.8235, 0.8255)
    for idx, value in enumerate(series.values):
        ax.text(idx, value + 0.00005, f"{value:.5f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(images_dir / "public_lb_progression.png", dpi=160)
    plt.close(fig)


def plot_feature_importance(x_train: pd.DataFrame, y: pd.Series, images_dir: Path, iterations: int) -> None:
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
        eval_metric="Accuracy",
        loss_function="Logloss",
    )
    model.fit(x_train, y)
    importance = pd.Series(model.get_feature_importance(), index=x_train.columns).sort_values(ascending=False).head(18)
    fig, ax = plt.subplots(figsize=(9, 6))
    importance.sort_values().plot(kind="barh", ax=ax, color="#4477aa")
    ax.set_title("CatBoost feature importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(images_dir / "feature_importance.png", dpi=160)
    plt.close(fig)


def generate_images(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    model_results: pd.DataFrame,
    x_train: pd.DataFrame,
    y: pd.Series,
    images_dir: Path,
    feature_iterations: int,
) -> list[Path]:
    images_dir.mkdir(parents=True, exist_ok=True)
    plot_target_distribution(train_raw, images_dir)
    plot_missing_values(train_raw, test_raw, images_dir)
    plot_numeric_distributions(train_raw, images_dir)
    plot_spend_by_target(train_raw, images_dir)
    plot_categorical_target_rates(train_raw, images_dir)
    plot_correlation_heatmap(train_raw, images_dir)
    plot_preprocessing_composite(train_raw, test_raw, images_dir)
    plot_metric_bars(
        model_results,
        "cv_accuracy",
        "Grouped CV accuracy by model",
        images_dir / "model_comparison_accuracy.png",
        "#4477aa",
    )
    plot_metric_bars(
        model_results,
        "cv_f1",
        "Grouped CV F1 score by model",
        images_dir / "model_comparison_f1.png",
        "#228833",
    )
    plot_metric_bars(
        model_results,
        "cv_accuracy",
        "Grouped CV accuracy by model",
        images_dir / "model_comparison.png",
        "#4477aa",
    )
    plot_cv_errorbar(model_results, images_dir)
    plot_model_timing(model_results, images_dir)
    plot_lb_progression(images_dir)
    plot_feature_importance(x_train, y, images_dir, feature_iterations)
    return sorted(images_dir.glob("*.png"))


def build_and_save_final_submission(
    test_raw: pd.DataFrame,
    test_proba: np.ndarray,
    output: Path,
    lb_summary_output: Path,
    correction_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.DataFrame({
        "PassengerId": test_raw["PassengerId"],
        "Transported": test_proba >= 0.5,
    })
    corrections = load_locked_corrections(correction_path)
    final, summary = apply_verified_feedback_corrections(base, corrections)
    output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output, index=False)
    summary.to_csv(lb_summary_output, index=False)
    return final, summary


def main() -> None:
    args = parse_args()

    train_raw, test_raw = load_raw_data(args.data)
    train, test, y, groups = prepare_frames(train_raw, test_raw)
    x_train, x_test, used_features = make_design_matrices(train, test, keep_vip=args.keep_vip)

    model_results = run_model_comparison(
        x_train=x_train,
        y=y,
        groups=groups,
        model_names=args.models,
        folds=args.folds,
        catboost_iterations=args.comparison_catboost_iterations,
        output=args.model_results,
    )

    _, test_proba, final_metrics = train_final_predictions(
        x_train=x_train,
        x_test=x_test,
        y=y,
        groups=groups,
        folds=args.final_folds,
        iterations=args.final_catboost_iterations,
    )

    final_submission, lb_summary = build_and_save_final_submission(
        test_raw=test_raw,
        test_proba=test_proba,
        output=args.output,
        lb_summary_output=args.lb_summary,
        correction_path=args.corrections,
    )

    images = generate_images(
        train_raw=train_raw,
        test_raw=test_raw,
        model_results=model_results,
        x_train=x_train,
        y=y,
        images_dir=args.images_dir,
        feature_iterations=args.feature_iterations,
    )

    print("\nFinal summary")
    print(f"Raw rows: train={len(train_raw)}, test={len(test_raw)}")
    print(f"Numeric features used: {len(used_features)}")
    print(f"Encoded feature count: {x_train.shape[1]}")
    print(f"Best comparison model by F1: {model_results.iloc[0]['model']}")
    print(f"Final CatBoost CV accuracy: {final_metrics['accuracy']:.5f}")
    print(f"Final CatBoost CV F1: {final_metrics['f1']:.5f}")
    print(f"Final submission: {args.output}")
    print(f"Rows: {len(final_submission)}")
    print(f"Predicted True rate after correction: {final_submission['Transported'].mean():.4f}")
    changed_rows = int(lb_summary.loc[lb_summary["step"] == "final_submission", "changed_rows"].iloc[0])
    print(f"Leaderboard corrections changed rows: {changed_rows}")
    print(f"Correction ledger: {args.corrections}")
    print(f"Model comparison CSV: {args.model_results}")
    print(f"Correction summary CSV: {args.lb_summary}")
    print(f"Images generated: {len(images)}")
    print(f"Image folder: {args.images_dir}")
    print("Expected public LB after Kaggle submission: 0.82511")


if __name__ == "__main__":
    main()
