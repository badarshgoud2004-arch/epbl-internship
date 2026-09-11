"""
Central configuration for the Healthcare Risk Prediction project.
Keeps dataset paths, column names, and constants in one place.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

# ---- Diabetes dataset (Pima Indians) ----
DIABETES_RAW_PATH = os.path.join(RAW_DIR, "diabetes.csv")
DIABETES_COLUMNS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age", "target"
]
# Columns where 0 is not physiologically valid -> treated as missing
DIABETES_ZERO_AS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]

# ---- Heart disease dataset (UCI Cleveland) ----
HEART_RAW_PATH = os.path.join(RAW_DIR, "heart.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.2

RISK_TIER_LABELS = ["Low", "Medium", "High"]
