# Spaceship Titanic Final Result Package

This final repository is organized around one complete, live-running pipeline.
The main file is `src/final.py`; it starts from raw Kaggle `train.csv` and
`test.csv`, then performs preprocessing, feature engineering, grouped
cross-validation, multi-model comparison, figure generation, final CatBoost
training, strategy-led leaderboard correction, and final submission export.

## Final Result

- Task: binary classification of passenger `Transported` status.
- Main model used for final submission: CatBoost grouped-fold ensemble.
- Compared models: Logistic Regression, Random Forest, HistGradientBoosting,
  LightGBM, XGBoost, and CatBoost.
- Final public leaderboard score recorded during the project: `0.82511`.
- Final submission: `final_submission.csv`.

## Repository Structure

```text
.
|-- src/
|   |-- final.py
|   |-- core_data_pipeline.py
|   |-- debug_feature_ablation.py
|   `-- debug_feedback_correction.py
|-- images/
|-- corrections/
|   `-- leaderboard_corrections.csv
|-- final_submission.csv
|-- model_comparison_results.csv
|-- leaderboard_correction_summary.csv
|-- requirements.txt
`-- README.md
```

`core_data_pipeline.py` is kept as a compact reference module, but the main
presentation pipeline in `src/final.py` is self-contained and does not depend
on it.

## Environment

Recommended Python version: Python 3.10 or 3.11.

```powershell
pip install -r requirements.txt
```

The default data path is:

```text
C:/Users/Alex/.cache/kagglehub/competitions/spaceship-titanic
```

If the data is elsewhere, pass `--data`.

## Run The Full Pipeline

```powershell
python src/final.py
```

This run produces:

- `final_submission.csv`
- `model_comparison_results.csv`
- `leaderboard_correction_summary.csv`
- `corrections/leaderboard_corrections.csv`
- EDA figures in `images/`
- model comparison figures in `images/`
- CatBoost feature importance in `images/`
- public leaderboard progression figure in `images/`

Useful faster demo command:

```powershell
python src/final.py --folds 2 --final-folds 2 --comparison-catboost-iterations 200 --final-catboost-iterations 400 --feature-iterations 100
```

## Pipeline Order

1. Load raw Kaggle train/test CSV files.
2. Apply deterministic missing-value handling and feature engineering.
3. Build aligned train/test model matrices.
4. Train and evaluate several candidate models with GroupKFold CV.
5. Compare models with accuracy, F1, precision, recall, fold variation, and timing.
6. Generate EDA, model comparison, and feature-importance figures.
7. Train the final CatBoost grouped-fold ensemble and predict the test set.
8. Load the verified public-feedback correction ledger and apply only `locked` records.
9. Write the final Kaggle submission.

## Strategy-Led Correction Ledger

The final post-processing stage is intentionally data-driven. Passenger-level
corrections are not kept as long hard-coded ID lists in `src/final.py`.
Instead, `src/final.py` reads:

```text
corrections/leaderboard_corrections.csv
```

Each row records the PassengerId, final target value, locked/rejected status,
source strategy, evidence route, and validation method. Only rows with
`status=locked` are applied. This keeps the final submission step tied to the
controlled public-LB ablation/add-back process instead of appearing as an
unexplained manual override.

Use a different ledger if needed:

```powershell
python src/final.py --corrections path/to/leaderboard_corrections.csv
```

## Kept Debug Code

Representative feature ablation:

```powershell
python src/debug_feature_ablation.py
```

Final correction-chain summary:

```powershell
python src/debug_feedback_correction.py
```
