# Spaceship Titanic ML Workshop Project

This repository contains the final reproducible code package for the AI3023
Machine Learning Workshop course project on the Kaggle Spaceship Titanic
competition.

## Project Summary

- Task: binary classification for passenger `Transported` status.
- Competition: Kaggle Spaceship Titanic.
- Final public LB used in this project: `0.82511`.
- Final submission file:
  `submissions/final/submission_082464_family_f2t_earth_es_pair.csv`.
- Main model backbone: CatBoost with compact feature engineering and
  group-aware validation.

## Repository Structure

```text
.
├── src/
│   ├── data_pipeline.py
│   ├── eda.py
│   ├── run_model_comparison.py
│   ├── run_hyperparameter_search.py
│   ├── catboost_baseline_model.py
│   └── demo_reproduce_final_submission.py
├── submissions/final/
├── reports/figures/
├── reports/tables/
├── docs/
├── requirements.txt
└── README.md
```

## Environment

Recommended Python version: Python 3.10 or 3.11.

```powershell
pip install -r requirements.txt
```

The scripts default to the local KaggleHub data path:

```text
C:/Users/Alex/.cache/kagglehub/competitions/spaceship-titanic
```

If your data is stored elsewhere, pass `--data` to the scripts.

## Reproduce Core Outputs

Generate EDA figures:

```powershell
python src/eda.py
```

Compare multiple ML models with 5-fold GroupKFold:

```powershell
python src/run_model_comparison.py --drop-vip
```

Run a compact CatBoost hyperparameter search:

```powershell
python src/run_hyperparameter_search.py --drop-vip
```

Train the clean CatBoost baseline:

```powershell
python src/catboost_baseline_model.py --drop-vip
```

Reproduce the final public-best submission from the saved baseline and the
validated correction chain:

```powershell
python src/demo_reproduce_final_submission.py
```

The final reproduction script verifies that the generated
`demo_final_submission.csv` matches the saved final submission.

## Key Outputs

- `reports/tables/model_comparison.csv`
- `reports/tables/catboost_hyperparameter_search.csv`
- `reports/figures/eda/`
- `reports/figures/feature_importance.png`
- `submissions/final/submission_082464_family_f2t_earth_es_pair.csv`

## Notes

Historical experiments and large submission archives are intentionally excluded
from this final repository to keep the submission package concise and
reproducible.
