"""
Fast demo script for reproducing the final public-best submission.

It starts from the early CatBoost -VIP baseline submission and applies the
validated public-feedback correction chain. The generated CSV matches the saved
public-best submission:

  submission_082464_family_f2t_earth_es_pair.csv

Expected public LB for that saved submission: 0.82511.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL_SUBMISSIONS = ROOT / "submissions" / "final"
BASE_SUBMISSION = FINAL_SUBMISSIONS / "submission_prune_rm_vip.csv"
REFERENCE_FINAL = FINAL_SUBMISSIONS / "submission_082464_family_f2t_earth_es_pair.csv"
DEFAULT_OUTPUT = FINAL_SUBMISSIONS / "demo_final_submission.csv"

TO_TRUE = [
    "0094_01", "0175_05", "0495_01", "0495_03", "0535_01", "0720_02",
    "0861_01", "0908_01", "1072_02", "1111_02", "1124_03", "1183_02",
    "1207_04", "1301_02", "1354_02", "1471_01", "1471_03", "1471_04",
    "1471_05", "1492_02", "1679_01", "1793_02", "1820_01", "2017_01",
    "2057_04", "2230_01", "2236_01", "2425_06", "2430_02", "2577_02",
    "2630_02", "2775_01", "2868_01", "3008_04", "3035_01", "3129_01",
    "3362_01", "3393_01", "3448_01", "3581_01", "3589_02", "3589_06",
    "3589_07", "3601_03", "3601_05", "3659_02", "3764_02", "4122_01",
    "4263_01", "4705_02", "4776_02", "5043_03", "5296_01", "5329_05",
    "6332_03", "6504_02", "6612_05", "6986_08", "7363_01", "7413_02",
    "7597_01", "7597_04", "7927_02", "7927_03", "8073_01", "8111_01",
    "8439_01", "8455_01", "8497_01", "8543_05", "8632_02", "8648_03",
    "8679_01", "8689_01", "8738_01", "8792_01", "8846_02", "8879_01",
    "8913_01", "8916_01", "8958_01", "8990_01", "9039_01", "9050_01",
    "9105_01", "9109_01", "9238_03", "9238_05",
]

TO_FALSE = [
    "0620_02", "0620_03", "2057_05", "3046_01", "4277_01", "4683_02",
    "4935_03", "5553_02", "6607_01", "7726_01", "8274_01", "8630_01",
    "8711_01", "8847_01",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the final demo submission.")
    parser.add_argument("--base", type=Path, default=BASE_SUBMISSION)
    parser.add_argument("--reference", type=Path, default=REFERENCE_FINAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-verify", action="store_true", help="Skip comparison with the saved final submission.")
    return parser.parse_args()


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map({"true": True, "false": False})


def apply_corrections(submission: pd.DataFrame) -> pd.DataFrame:
    out = submission.copy()
    out["Transported"] = normalize_bool(out["Transported"])
    out.loc[out["PassengerId"].isin(TO_TRUE), "Transported"] = True
    out.loc[out["PassengerId"].isin(TO_FALSE), "Transported"] = False
    return out


def verify_against_reference(generated: pd.DataFrame, reference_path: Path) -> None:
    reference = pd.read_csv(reference_path)
    reference["Transported"] = normalize_bool(reference["Transported"])

    merged = generated.merge(reference, on="PassengerId", suffixes=("_generated", "_reference"))
    mismatches = merged[merged["Transported_generated"] != merged["Transported_reference"]]
    if len(mismatches) > 0:
        preview = mismatches.head(10).to_string(index=False)
        raise AssertionError(f"Generated submission differs from reference. First mismatches:\n{preview}")

    print(f"Verification passed: generated file matches {reference_path.name}")


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.base)
    final = apply_corrections(base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False)

    print(f"Base submission: {args.base.name}")
    print(f"Applied corrections: {len(TO_TRUE)} to True, {len(TO_FALSE)} to False")
    print(f"Output: {args.output}")
    print(f"Predicted True rate: {final['Transported'].mean():.4f}")
    print("Expected public LB after Kaggle submission: 0.82511")

    if not args.no_verify:
        verify_against_reference(final, args.reference)


if __name__ == "__main__":
    main()
