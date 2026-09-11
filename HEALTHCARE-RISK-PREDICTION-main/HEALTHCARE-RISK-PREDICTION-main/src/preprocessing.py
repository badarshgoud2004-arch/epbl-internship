"""
Data loading and preprocessing for the diabetes and heart disease datasets.
Handles missing-value imputation, scaling, and class-imbalance correction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import os

import config


def load_diabetes():
    df = pd.read_csv(config.DIABETES_RAW_PATH, header=None, names=config.DIABETES_COLUMNS)
    # Zeros in these columns are biologically impossible -> treat as missing
    for col in config.DIABETES_ZERO_AS_MISSING:
        df[col] = df[col].replace(0, np.nan)
    return df


def load_heart():
    df = pd.read_csv(config.HEART_RAW_PATH)
    df = df.rename(columns={"target": "target"})
    return df


def _split_and_scale(df, target_col, apply_smote=True):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    # Impute missing values (median strategy)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X.columns, index=X_test.index)

    if apply_smote:
        sm = SMOTE(random_state=config.RANDOM_STATE)
        X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)

    return {
        "X_train": X_train_scaled, "X_test": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "imputer": imputer, "scaler": scaler,
        "feature_names": list(X.columns),
        # unscaled test set kept for readable SHAP / what-if displays
        "X_test_raw": X_test_imp,
    }


def prepare_dataset(name):
    """name: 'diabetes' or 'heart'. Returns dict of train/test splits + preprocessing objects."""
    if name == "diabetes":
        df = load_diabetes()
        bundle = _split_and_scale(df, "target")
    elif name == "heart":
        df = load_heart()
        bundle = _split_and_scale(df, "target")
    else:
        raise ValueError("name must be 'diabetes' or 'heart'")

    bundle["raw_df"] = df
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    df.to_csv(os.path.join(config.PROCESSED_DIR, f"{name}_clean.csv"), index=False)
    return bundle


if __name__ == "__main__":
    for name in ["diabetes", "heart"]:
        b = prepare_dataset(name)
        print(f"{name}: train={b['X_train'].shape}, test={b['X_test'].shape}, "
              f"pos_rate_train={b['y_train'].mean():.2f}, pos_rate_test={b['y_test'].mean():.2f}")
