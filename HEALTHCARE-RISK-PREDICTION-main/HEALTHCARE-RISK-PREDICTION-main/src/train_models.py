"""
Trains and compares multiple classical ML models (Logistic Regression, Random Forest,
XGBoost, SVM) on each dataset, tunes hyperparameters, and saves the best model per dataset.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config
from preprocessing import prepare_dataset

MODEL_GRID = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
        "params": {"C": [0.01, 0.1, 1, 10]},
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=config.RANDOM_STATE),
        "params": {"n_estimators": [200, 400], "max_depth": [4, 6, None]},
    },
    "xgboost": {
        "estimator": XGBClassifier(
            random_state=config.RANDOM_STATE, eval_metric="logloss", use_label_encoder=False
        ),
        "params": {"n_estimators": [200, 400], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
    },
    "svm": {
        "estimator": SVC(probability=True, random_state=config.RANDOM_STATE),
        "params": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    },
}


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }


def train_all_models(dataset_name):
    bundle = prepare_dataset(dataset_name)
    X_train, X_test = bundle["X_train"], bundle["X_test"]
    y_train, y_test = bundle["y_train"], bundle["y_test"]

    results = {}
    fitted_models = {}

    for model_name, spec in MODEL_GRID.items():
        grid = GridSearchCV(
            spec["estimator"], spec["params"], cv=5, scoring="roc_auc", n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        metrics = evaluate(best_model, X_test, y_test)
        metrics["best_params"] = grid.best_params_
        results[model_name] = metrics
        fitted_models[model_name] = best_model
        print(f"[{dataset_name}] {model_name}: {metrics}")

    # pick best model by roc_auc
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = fitted_models[best_name]
    print(f"[{dataset_name}] BEST MODEL -> {best_name} (roc_auc={results[best_name]['roc_auc']})")

    # Save all artifacts
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(config.MODELS_DIR, f"{dataset_name}_best_model.joblib"))
    joblib.dump(bundle["scaler"], os.path.join(config.MODELS_DIR, f"{dataset_name}_scaler.joblib"))
    joblib.dump(bundle["imputer"], os.path.join(config.MODELS_DIR, f"{dataset_name}_imputer.joblib"))
    joblib.dump(fitted_models, os.path.join(config.MODELS_DIR, f"{dataset_name}_all_models.joblib"))

    with open(os.path.join(config.MODELS_DIR, f"{dataset_name}_results.json"), "w") as f:
        json.dump({"results": results, "best_model": best_name,
                   "feature_names": bundle["feature_names"]}, f, indent=2)

    # Confusion matrix plot for best model
    out_dir = os.path.join(config.FIGURES_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{dataset_name.capitalize()}: Confusion Matrix ({best_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=120)
    plt.close()

    # Model comparison bar chart
    metrics_df = pd.DataFrame({m: {k: v for k, v in r.items() if k != "best_params"}
                                for m, r in results.items()}).T
    metrics_df[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(
        kind="bar", figsize=(9, 5), colormap="viridis"
    )
    plt.title(f"{dataset_name.capitalize()}: Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=20)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "model_comparison.png"), dpi=120)
    plt.close()

    return results, best_name, best_model, bundle


if __name__ == "__main__":
    for ds in ["diabetes", "heart"]:
        train_all_models(ds)
