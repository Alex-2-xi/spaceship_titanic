"""Compare multiple machine learning models with GroupKFold validation."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_pipeline import DEFAULT_DATA_PATH, ROOT, load_prepared_matrices


DEFAULT_OUT_DIR = ROOT / "reports" / "tables"
DEFAULT_MODELS = ["logistic", "random_forest", "histgb", "lightgbm", "xgboost", "catboost"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model comparison with grouped cross-validation.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--drop-vip", action="store_true", help="Use the stronger no-VIP feature set.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS,
        help="Subset of models to run.",
    )
    return parser.parse_args()


def build_model(name: str, seed: int = 42):
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
            iterations=1200,
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


def predict_positive_proba(model, x_valid: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_valid)[:, 1]
    return model.predict(x_valid).astype(float)


def evaluate_model(
    name: str,
    x: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    folds: int,
) -> dict[str, object]:
    gkf = GroupKFold(n_splits=folds)
    oof = np.zeros(len(x))
    fold_scores: list[float] = []
    train_seconds = 0.0
    inference_seconds = 0.0

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(x, y, groups=groups), start=1):
        model = build_model(name, seed=42 + fold)
        start = time.perf_counter()
        model.fit(x.iloc[tr_idx], y.iloc[tr_idx])
        train_seconds += time.perf_counter() - start

        start = time.perf_counter()
        oof[val_idx] = predict_positive_proba(model, x.iloc[val_idx])
        inference_seconds += time.perf_counter() - start

        fold_acc = accuracy_score(y.iloc[val_idx], oof[val_idx] >= 0.5)
        fold_scores.append(fold_acc)
        print(f"{name} fold={fold} accuracy={fold_acc:.4f}")

    return {
        "model": name,
        "folds": folds,
        "cv_accuracy": accuracy_score(y, oof >= 0.5),
        "fold_mean_accuracy": float(np.mean(fold_scores)),
        "fold_std_accuracy": float(np.std(fold_scores)),
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "suitability_note": suitability_note(name),
    }


def suitability_note(name: str) -> str:
    notes = {
        "logistic": "Fast linear baseline; limited nonlinear capacity.",
        "random_forest": "Robust tree ensemble; weaker on this small sparse encoded dataset.",
        "histgb": "Efficient gradient boosting baseline from scikit-learn.",
        "lightgbm": "Fast boosted trees; less stable than CatBoost in prior experiments.",
        "xgboost": "Strong boosted-tree baseline; needs careful regularization.",
        "catboost": "Best project backbone; stable on mixed categorical/numeric features.",
    }
    return notes[name]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    x_train, _, y, groups, _, features = load_prepared_matrices(args.data, drop_vip=args.drop_vip)

    print(f"Rows={len(x_train)}, encoded_features={x_train.shape[1]}, numeric_features={len(features)}")
    print(f"Models={', '.join(args.models)}, folds={args.folds}, drop_vip={args.drop_vip}")

    rows = [evaluate_model(name, x_train, y, groups, args.folds) for name in args.models]
    result = pd.DataFrame(rows).sort_values("cv_accuracy", ascending=False)
    output = args.out_dir / "model_comparison.csv"
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
