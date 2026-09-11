"""
Converts binary risk probabilities into clinically-intuitive Low/Medium/High
risk tiers using fixed probability thresholds, and reports tier distribution.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd

import config
from preprocessing import prepare_dataset

# Probability cut points -> tier. Adjustable per clinical preference.
THRESHOLDS = {"low_max": 0.33, "medium_max": 0.66}


def stratify(prob):
    if prob < THRESHOLDS["low_max"]:
        return "Low"
    elif prob < THRESHOLDS["medium_max"]:
        return "Medium"
    else:
        return "High"


def run_stratification(dataset_name):
    bundle = prepare_dataset(dataset_name)
    model = joblib.load(os.path.join(config.MODELS_DIR, f"{dataset_name}_best_model.joblib"))

    X_test, y_test = bundle["X_test"], bundle["y_test"]
    probs = model.predict_proba(X_test)[:, 1]
    tiers = pd.Series([stratify(p) for p in probs], index=X_test.index, name="risk_tier")

    out = pd.DataFrame({
        "actual": y_test.values,
        "predicted_probability": probs.round(3),
        "risk_tier": tiers.values,
    })
    out_path = os.path.join(config.MODELS_DIR, f"{dataset_name}_risk_tiers.csv")
    out.to_csv(out_path, index=False)

    tier_counts = out["risk_tier"].value_counts().reindex(config.RISK_TIER_LABELS, fill_value=0)
    # actual positive rate within each tier -> sanity check that tiers are meaningful
    tier_actual_rate = out.groupby("risk_tier")["actual"].mean().reindex(config.RISK_TIER_LABELS)

    summary = {
        "thresholds": THRESHOLDS,
        "tier_counts": tier_counts.to_dict(),
        "actual_positive_rate_by_tier": tier_actual_rate.round(2).to_dict(),
    }
    with open(os.path.join(config.MODELS_DIR, f"{dataset_name}_risk_tier_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{dataset_name}] risk tier counts: {tier_counts.to_dict()}")
    print(f"[{dataset_name}] actual positive rate by tier: {tier_actual_rate.round(2).to_dict()}")
    return out, summary


if __name__ == "__main__":
    for ds in ["diabetes", "heart"]:
        run_stratification(ds)
