"""
Runs the entire pipeline end-to-end: EDA -> preprocessing -> model training
-> explainability -> risk stratification -> fairness check.
Usage: python run_pipeline.py
"""
from eda import run_eda
from preprocessing import load_diabetes, load_heart
from train_models import train_all_models
from explainability import explain_dataset
from risk_stratification import run_stratification
from fairness_check import run_fairness_check

DATASETS = ["diabetes", "heart"]
LOADERS = {"diabetes": load_diabetes, "heart": load_heart}


def main():
    print("=" * 60, "\nSTEP 1: EDA\n", "=" * 60)
    for ds in DATASETS:
        run_eda(LOADERS[ds](), ds)

    print("=" * 60, "\nSTEP 2: MODEL TRAINING & COMPARISON\n", "=" * 60)
    for ds in DATASETS:
        train_all_models(ds)

    print("=" * 60, "\nSTEP 3: EXPLAINABILITY (SHAP)\n", "=" * 60)
    for ds in DATASETS:
        explain_dataset(ds)

    print("=" * 60, "\nSTEP 4: RISK STRATIFICATION\n", "=" * 60)
    for ds in DATASETS:
        run_stratification(ds)

    print("=" * 60, "\nSTEP 5: FAIRNESS CHECK\n", "=" * 60)
    for ds in DATASETS:
        run_fairness_check(ds)

    print("\nPipeline complete. Artifacts saved in /models and /reports/figures.")


if __name__ == "__main__":
    main()
