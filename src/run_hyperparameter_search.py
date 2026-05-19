"""Run a compact CatBoost hyperparameter search with GroupKFold validation."""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from data_pipeline import DEFAULT_DATA_PATH, ROOT, load_prepared_matrices


DEFAULT_OUT_DIR = ROOT / "reports" / "tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search CatBoost hyperparameters with GroupKFold.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--drop-vip", action="store_true", help="Search with the no-VIP feature set.")
    parser.add_argument("--full", action="store_true", help="Use a larger grid.")
    return parser.parse_args()


def param_grid(full: bool) -> list[dict[str, float | int]]:
    if full:
        depths = [6, 8, 10]
        learning_rates = [0.03, 0.05]
        l2_values = [3, 5, 10]
        iterations = [1500]
    else:
        depths = [6, 8]
        learning_rates = [0.05]
        l2_values = [5, 10]
        iterations = [800]

    keys = ["depth", "learning_rate", "l2_leaf_reg", "iterations"]
    return [dict(zip(keys, values)) for values in itertools.product(depths, learning_rates, l2_values, iterations)]


def evaluate_params(
    params: dict[str, float | int],
    x: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    folds: int,
) -> dict[str, object]:
    gkf = GroupKFold(n_splits=folds)
    oof = np.zeros(len(x))
    fold_scores: list[float] = []
    start = time.perf_counter()

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(x, y, groups=groups), start=1):
        model = CatBoostClassifier(
            **params,
            random_seed=42 + fold,
            verbose=0,
            allow_writing_files=False,
            early_stopping_rounds=80,
            eval_metric="Accuracy",
            loss_function="Logloss",
        )
        model.fit(
            x.iloc[tr_idx],
            y.iloc[tr_idx],
            eval_set=(x.iloc[val_idx], y.iloc[val_idx]),
            verbose=0,
        )
        oof[val_idx] = model.predict_proba(x.iloc[val_idx])[:, 1]
        fold_acc = accuracy_score(y.iloc[val_idx], oof[val_idx] >= 0.5)
        fold_scores.append(fold_acc)
        print(f"params={params} fold={fold} accuracy={fold_acc:.4f}")

    elapsed = time.perf_counter() - start
    return {
        **params,
        "folds": folds,
        "cv_accuracy": accuracy_score(y, oof >= 0.5),
        "fold_mean_accuracy": float(np.mean(fold_scores)),
        "fold_std_accuracy": float(np.std(fold_scores)),
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    x_train, _, y, groups, _, features = load_prepared_matrices(args.data, drop_vip=args.drop_vip)

    grid = param_grid(args.full)
    print(f"Rows={len(x_train)}, encoded_features={x_train.shape[1]}, numeric_features={len(features)}")
    print(f"Grid size={len(grid)}, folds={args.folds}, drop_vip={args.drop_vip}, full={args.full}")

    rows = [evaluate_params(params, x_train, y, groups, args.folds) for params in grid]
    result = pd.DataFrame(rows).sort_values("cv_accuracy", ascending=False)
    output = args.out_dir / "catboost_hyperparameter_search.csv"
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
