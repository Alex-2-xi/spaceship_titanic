from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from core_data_pipeline import (
    BASE_NUMERIC,
    DEFAULT_DATA_PATH,
    ROOT,
    load_raw_data,
    make_design_matrices,
    prepare_frames,
)


DEFAULT_OUTPUT = ROOT / "debug_feature_ablation_results.csv"
ABLATIONS = {
    "final_numeric_set": [],
    "keep_vip": ["__keep_vip__"],
    "drop_spend_logs": [
        "RoomService_log", "FoodCourt_log", "ShoppingMall_log", "Spa_log", "VRDeck_log",
        "TotalSpend_log", "LuxurySpend_log", "SpendCount", "NoSpend",
    ],
    "drop_group_features": ["GroupSize", "IsAlone"],
    "drop_child_cryo_features": ["IsChild", "IsChildOrCryo"],
    "drop_cabin_number_features": ["CabinNum", "CabinNumBin"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Representative feature ablation for the final CatBoost model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=800)
    return parser.parse_args()


def build_model(iterations: int, seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=5,
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
        early_stopping_rounds=80,
        eval_metric="Accuracy",
        loss_function="Logloss",
    )


def evaluate(x: pd.DataFrame, y: pd.Series, groups: np.ndarray, folds: int, iterations: int) -> float:
    gkf = GroupKFold(n_splits=folds)
    oof = np.zeros(len(x))
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(x, y, groups=groups), start=1):
        model = build_model(iterations, seed=42 + fold)
        model.fit(x.iloc[tr_idx], y.iloc[tr_idx], eval_set=(x.iloc[val_idx], y.iloc[val_idx]), verbose=0)
        oof[val_idx] = model.predict_proba(x.iloc[val_idx])[:, 1]
    return accuracy_score(y, oof >= 0.5)


def feature_set_for_ablation(name: str, removed: list[str]) -> tuple[list[str], bool]:
    if name == "keep_vip":
        return list(BASE_NUMERIC), False
    features = [feature for feature in BASE_NUMERIC if feature != "VIP" and feature not in removed]
    return features, True


def main() -> None:
    args = parse_args()
    train_raw, test_raw = load_raw_data(args.data)
    train, test, y, groups = prepare_frames(train_raw, test_raw)

    rows = []
    for name, removed in ABLATIONS.items():
        numeric_features, drop_vip = feature_set_for_ablation(name, removed)
        x_train, _, used_features = make_design_matrices(
            train,
            test,
            drop_vip=drop_vip,
            numeric_features=numeric_features,
        )
        cv_accuracy = evaluate(x_train, y, groups, args.folds, args.iterations)
        rows.append({
            "ablation": name,
            "removed_features": ";".join(feature for feature in removed if feature != "__keep_vip__"),
            "numeric_features": len(used_features),
            "encoded_features": x_train.shape[1],
            "folds": args.folds,
            "cv_accuracy": cv_accuracy,
        })
        print(f"{name}: cv_accuracy={cv_accuracy:.5f}")

    result = pd.DataFrame(rows).sort_values("cv_accuracy", ascending=False)
    result.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
