"""Train the clean CatBoost baseline and write a Kaggle submission file.

Usage:
  python src/catboost_baseline_model.py
  python src/catboost_baseline_model.py --drop-vip
  python src/catboost_baseline_model.py --drop-vip --full
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from data_pipeline import DEFAULT_DATA_PATH, ROOT, load_prepared_matrices


OUT_DIR = ROOT / "submissions" / "final"
SEEDS_QUICK = [42]
SEEDS_FULL = [42, 123, 456, 789, 2024, 7, 99, 2023, 314, 1618]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the clean CatBoost baseline.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--full", action="store_true", help="Use 10 seeds instead of the quick 1-seed run.")
    parser.add_argument("--drop-vip", action="store_true", help="Remove VIP from numeric features.")
    parser.add_argument("--output", default=None, help="Submission filename. Defaults depend on --drop-vip.")
    return parser.parse_args()


def build_catboost(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=5000,
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


def train_baseline(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray, float]:
    all_oof = np.zeros(len(x_train))
    all_test_proba = np.zeros(len(x_test))

    for seed in seeds:
        oof = np.zeros(len(x_train))
        test_proba = np.zeros(len(x_test))
        gkf = GroupKFold(n_splits=10)

        for fold, (tr_idx, val_idx) in enumerate(gkf.split(x_train, y, groups=groups), start=1):
            model = build_catboost(seed)
            model.fit(
                x_train.iloc[tr_idx],
                y.iloc[tr_idx],
                eval_set=(x_train.iloc[val_idx], y.iloc[val_idx]),
                verbose=0,
            )
            oof[val_idx] = model.predict_proba(x_train.iloc[val_idx])[:, 1]
            test_proba += model.predict_proba(x_test)[:, 1] / gkf.n_splits
            fold_score = accuracy_score(y.iloc[val_idx], (oof[val_idx] > 0.5).astype(int))
            print(f"seed={seed} fold={fold} acc={fold_score:.4f}")

        all_oof += oof / len(seeds)
        all_test_proba += test_proba / len(seeds)

    cv = accuracy_score(y, (all_oof > 0.5).astype(int))
    return all_oof, all_test_proba, cv


def main() -> None:
    args = parse_args()
    x_train, x_test, y, groups, passenger_ids, features = load_prepared_matrices(
        args.data,
        drop_vip=args.drop_vip,
    )
    seeds = SEEDS_FULL if args.full else SEEDS_QUICK

    print(f"Rows: train={len(x_train)}, test={len(x_test)}")
    print(f"Features: numeric={len(features)}, encoded={x_train.shape[1]}, seeds={len(seeds)}")
    print(f"VIP included: {'VIP' in features}")

    _, test_proba, cv = train_baseline(x_train, x_test, y, groups, seeds)
    preds = test_proba >= 0.5

    args.out_dir.mkdir(parents=True, exist_ok=True)
    default_name = "submission_catboost_baseline_drop_vip.csv" if args.drop_vip else "submission_catboost_baseline.csv"
    output_name = args.output or default_name
    output_path = args.out_dir / output_name
    pd.DataFrame({"PassengerId": passenger_ids, "Transported": preds}).to_csv(output_path, index=False)

    print(f"CV accuracy: {cv:.4f}")
    print(f"Predicted True: {preds.sum()} / {len(preds)} ({preds.mean():.1%})")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
