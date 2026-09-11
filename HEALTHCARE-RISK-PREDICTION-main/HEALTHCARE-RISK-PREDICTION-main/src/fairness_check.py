"""
Basic fairness/bias audit: compares model performance (accuracy, recall, FNR)
across demographic subgroups (age bands, and sex where available) to check
whether the model systematically underperforms for any group.
Important in healthcare ML: a model with high overall accuracy can still
silently fail a specific subgroup, which is a real deployment risk.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

import config
from preprocessing import prepare_dataset


def _age_band(age):
    if age < 35:
        return "under_35"
    elif age < 55:
        return "35_54"
    else:
        return "55_plus"


def run_fairness_check(dataset_name):
    bundle = prepare_dataset(dataset_name)
    model = joblib.load(os.path.join(config.MODELS_DIR, f"{dataset_name}_best_model.joblib"))

    X_test, y_test = bundle["X_test"], bundle["y_test"]
    X_test_raw = bundle["X_test_raw"]  # unscaled, for readable age/sex grouping
    y_pred = model.predict(X_test)

    df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred}, index=X_test.index)
    df["age"] = X_test_raw["age"].values
    df["age_band"] = df["age"].apply(_age_band)

    groups = {"age_band": df["age_band"]}
    if "sex" in X_test_raw.columns:
        sex_map = {0: "female", 1: "male"}
        df["sex"] = X_test_raw["sex"].map(sex_map).fillna(X_test_raw["sex"])
        groups["sex"] = df["sex"]

    report = {}
    for group_name, group_series in groups.items():
        report[group_name] = {}
        for g in sorted(group_series.unique(), key=str):
            mask = (group_series == g)
            if mask.sum() < 5:
                continue  # skip groups too small to be meaningful
            sub = df[mask]
            acc = accuracy_score(sub["y_true"], sub["y_pred"])
            # False Negative Rate: proportion of actual positives the model misses
            # (most clinically costly error type -> highlighted per subgroup)
            actual_pos = sub[sub["y_true"] == 1]
            fnr = (actual_pos["y_pred"] == 0).mean() if len(actual_pos) > 0 else None
            report[group_name][str(g)] = {
                "n": int(mask.sum()),
                "accuracy": round(acc, 3),
                "false_negative_rate": round(fnr, 3) if fnr is not None else None,
            }

    out_path = os.path.join(config.MODELS_DIR, f"{dataset_name}_fairness_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[{dataset_name}] fairness report:")
    print(json.dumps(report, indent=2))

    # Flag largest accuracy gap within each group type
    flags = []
    for group_name, subgroups in report.items():
        accs = {k: v["accuracy"] for k, v in subgroups.items()}
        if len(accs) > 1:
            gap = max(accs.values()) - min(accs.values())
            if gap > 0.15:
                flags.append(f"{group_name}: accuracy gap of {gap:.2f} across subgroups ({accs})")
    if flags:
        print(f"[{dataset_name}] FAIRNESS FLAGS: {flags}")
    else:
        print(f"[{dataset_name}] No large accuracy gaps (>0.15) detected across subgroups.")

    return report, flags


if __name__ == "__main__":
    for ds in ["diabetes", "heart"]:
        run_fairness_check(ds)
        print()
