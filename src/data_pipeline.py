"""Shared data preparation utilities for the Spaceship Titanic project."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = Path("C:/Users/Alex/.cache/kagglehub/competitions/spaceship-titanic")
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CAT_COLS = ["HomePlanet", "Destination", "Deck", "Side"]
BASE_NUMERIC = [
    "CryoSleep", "Age", "VIP",
    "RoomService_log", "FoodCourt_log", "ShoppingMall_log", "Spa_log", "VRDeck_log",
    "TotalSpend_log", "NoSpend", "CabinNum", "GroupSize", "IsAlone",
    "SpendCount", "LuxurySpend_log", "IsChild", "IsChildOrCryo", "CabinNumBin",
]


def load_raw_data(data_dir: Path = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Kaggle train/test CSV files from the configured data directory."""
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected train.csv and test.csv under {data_dir}. "
            "Download the Kaggle Spaceship Titanic data or pass --data."
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def compute_fill_values(train_df: pd.DataFrame) -> dict[str, object]:
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
    fill_values: dict[str, object],
    surname_map: dict[str, str],
) -> pd.DataFrame:
    """Apply the compact, leakage-aware imputation chain used by the final model."""
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
    """Return feature-engineered train/test frames, target, and passenger groups."""
    fill_values = compute_fill_values(train_raw)
    surname_map = build_surname_planet_map(train_raw)
    train = engineer_features(intelligent_impute(train_raw, fill_values, surname_map))
    test = engineer_features(intelligent_impute(test_raw, fill_values, surname_map))
    y = train_raw["Transported"].astype(int)
    groups = train["Group"].astype(int).to_numpy()
    return train, test, y, groups


def make_design_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    drop_vip: bool = False,
    numeric_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build aligned one-hot encoded model matrices for train and test."""
    features = list(numeric_features or BASE_NUMERIC)
    if drop_vip:
        features = [feature for feature in features if feature != "VIP"]

    train_enc = pd.get_dummies(train[features + CAT_COLS], columns=CAT_COLS)
    test_enc = pd.get_dummies(test[features + CAT_COLS], columns=CAT_COLS)
    x_train, x_test = train_enc.align(test_enc, join="left", axis=1, fill_value=0)
    if "Deck_T" in x_train.columns:
        x_train = x_train.drop(columns=["Deck_T"])
        x_test = x_test.drop(columns=["Deck_T"])
    return x_train, x_test, features


def load_prepared_matrices(
    data_dir: Path = DEFAULT_DATA_PATH,
    drop_vip: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, np.ndarray, pd.Series, list[str]]:
    """Load raw data and return train/test matrices plus metadata."""
    train_raw, test_raw = load_raw_data(data_dir)
    train, test, y, groups = prepare_frames(train_raw, test_raw)
    x_train, x_test, features = make_design_matrices(train, test, drop_vip=drop_vip)
    return x_train, x_test, y, groups, test_raw["PassengerId"], features
