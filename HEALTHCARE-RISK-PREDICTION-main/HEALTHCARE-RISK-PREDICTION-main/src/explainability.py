"""
SHAP-based explainability layer.
Generates global (summary) and local (individual prediction) explanations
for the best model of each dataset, and cross-checks top features informally
against known medical risk factors.
"""
import os
import json
import joblib
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from preprocessing import prepare_dataset

# Informal cross-reference: features widely reported in medical literature as
# major risk factors, used only to comment on agreement in the printed summary.
KNOWN_RISK_FACTORS = {
    "diabetes": ["glucose", "bmi", "age", "diabetes_pedigree"],
    "heart": ["cp", "thalach", "oldpeak", "ca", "thal", "age"],
}


def explain_dataset(dataset_name):
    bundle = prepare_dataset(dataset_name)
    model = joblib.load(os.path.join(config.MODELS_DIR, f"{dataset_name}_best_model.joblib"))
    X_test = bundle["X_test"]
    feature_names = bundle["feature_names"]

    out_dir = os.path.join(config.FIGURES_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # Use TreeExplainer for tree models, fallback to KernelExplainer otherwise
    model_type = type(model).__name__
    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):  # older shap: list [class0, class1]
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # newer shap: shape (n_samples, n_features, n_classes)
            shap_values = shap_values[:, :, 1]
    else:
        background = shap.sample(bundle["X_train"], 50, random_state=config.RANDOM_STATE)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test, nsamples=100)[1]

    # --- Global explanation: summary plot ---
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # Mean absolute SHAP value ranking (global feature importance)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_rank = sorted(
        zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True
    )
    top_features = [f for f, _ in importance_rank[:5]]

    # --- Local explanation: waterfall for a couple of individual patients ---
    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            ev_arr = np.atleast_1d(expected_value)
            expected_value = ev_arr[1] if len(ev_arr) > 1 else ev_arr[0]
    else:
        expected_value = explainer.expected_value[1]

    for i in [0, 1]:
        plt.figure()
        exp = shap.Explanation(
            values=shap_values[i], base_values=expected_value,
            data=X_test.iloc[i].values, feature_names=feature_names
        )
        shap.plots.waterfall(exp, show=False, max_display=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"shap_waterfall_patient{i}.png"), dpi=120, bbox_inches="tight")
        plt.close()

    # --- Save importance ranking + literature cross-check ---
    known = set(KNOWN_RISK_FACTORS.get(dataset_name, []))
    overlap = [f for f in top_features if f in known]

    result = {
        "top_features_by_shap": top_features,
        "known_literature_risk_factors": list(known),
        "overlap_with_literature": overlap,
        "agreement_ratio": round(len(overlap) / max(len(known), 1), 2),
    }
    with open(os.path.join(config.MODELS_DIR, f"{dataset_name}_shap_summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"[{dataset_name}] top SHAP features: {top_features}")
    print(f"[{dataset_name}] overlap with known literature risk factors: {overlap} "
          f"({result['agreement_ratio']*100:.0f}% agreement)")

    return result, explainer, shap_values, bundle


if __name__ == "__main__":
    for ds in ["diabetes", "heart"]:
        explain_dataset(ds)
