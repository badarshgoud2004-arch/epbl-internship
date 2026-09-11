# Healthcare Risk Prediction System with Explainability

A capstone data science project that predicts a patient's risk of **diabetes** or
**heart disease** from clinical data, using classical ML — and explains *why* each
prediction was made using SHAP. Goes beyond a black-box classifier by adding risk
tiers, a live what-if simulator, and a subgroup fairness audit.

## Why this project

Most "disease prediction" student projects stop at reporting accuracy. This one
answers the question a doctor actually asks: **"Which patients are at risk, why,
and can I trust this model equally for everyone?"**

Real-world relevance:
- Clinical decision support (second-opinion tool for doctors)
- Preventive care / insurance risk screening
- Telemedicine triage in resource-limited settings
- Patient-facing health apps that explain risk factors, not just a number

## Project Structure

```
healthcare-risk-prediction/
├── data/
│   ├── raw/                  # Original datasets (Pima diabetes, UCI heart disease)
│   └── processed/            # Cleaned datasets after preprocessing
├── src/
│   ├── config.py             # Paths, constants, column definitions
│   ├── preprocessing.py      # Missing value imputation, scaling, SMOTE
│   ├── eda.py                # Exploratory data analysis + plots
│   ├── train_models.py       # Trains & compares 4 classifiers per dataset
│   ├── explainability.py     # SHAP global/local explanations
│   ├── risk_stratification.py # Converts probability -> Low/Medium/High tier
│   ├── fairness_check.py     # Subgroup performance audit (age, sex)
│   └── run_pipeline.py       # Runs the entire pipeline end-to-end
├── app/
│   └── streamlit_app.py      # Interactive demo: prediction, what-if, comparison
├── models/                   # Saved trained models + metrics (generated)
├── reports/figures/          # Generated EDA & SHAP plots (generated)
├── requirements.txt
└── README.md
```

## Datasets

| Dataset | Source | Rows | Features | Target |
|---|---|---|---|---|
| Diabetes | Pima Indians Diabetes Dataset | 768 | 8 | Diabetic (1) / Not (0) |
| Heart Disease | UCI Cleveland Heart Disease | 303 | 13 | Disease present (1) / Not (0) |

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (EDA -> training -> SHAP -> risk tiers -> fairness)
cd src
python run_pipeline.py

# 3. Launch the interactive demo app
cd ..
streamlit run app/streamlit_app.py
```

Each stage can also be run individually, e.g. `python src/train_models.py`.

## Methodology

1. **EDA**: missing-value analysis, class balance, correlation heatmaps, feature
   distributions split by outcome (`src/eda.py`)
2. **Preprocessing**: median imputation (biologically-impossible zeros in the
   diabetes dataset are treated as missing), standard scaling, SMOTE oversampling
   on the training set only to correct class imbalance (`src/preprocessing.py`)
3. **Modeling**: four classical models — Logistic Regression, Random Forest,
   XGBoost, SVM — each tuned via `GridSearchCV` (5-fold CV, ROC-AUC scoring)
   (`src/train_models.py`)
4. **Explainability**: SHAP `TreeExplainer` for tree models, global summary plots
   plus per-patient waterfall plots for individual predictions; top SHAP features
   cross-checked against known clinical risk factors (`src/explainability.py`)
5. **Risk stratification**: predicted probabilities bucketed into Low
   (<33%) / Medium (33–66%) / High (>66%) tiers, validated by checking that
   actual outcome rate increases monotonically across tiers (`src/risk_stratification.py`)
6. **Fairness check**: accuracy and false-negative rate compared across age
   bands (and sex, for the heart dataset) to catch subgroup performance gaps
   before considering any deployment (`src/fairness_check.py`)

## Results

### Diabetes — model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.714 | 0.583 | 0.648 | 0.614 | 0.808 |
| Random Forest | 0.727 | 0.594 | 0.704 | 0.644 | 0.815 |
| **XGBoost (best)** | **0.753** | **0.633** | **0.704** | **0.667** | **0.822** |
| SVM | 0.727 | 0.597 | 0.685 | 0.638 | 0.811 |

### Heart Disease — model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.803 | 0.769 | 0.909 | 0.833 | 0.876 |
| **Random Forest (best)** | **0.820** | **0.775** | **0.939** | **0.849** | **0.913** |
| XGBoost | 0.754 | 0.737 | 0.848 | 0.789 | 0.858 |
| SVM | 0.803 | 0.818 | 0.818 | 0.818 | 0.869 |

### Explainability findings

- **Diabetes**: top SHAP-ranked features were `glucose`, `bmi`, `age`,
  `diabetes_pedigree`, `insulin` — **100% overlap** with established clinical
  risk factors for Type 2 diabetes.
- **Heart Disease**: top SHAP-ranked features were `cp` (chest pain type),
  `thal`, `ca`, `oldpeak`, `exang` — **67% overlap** with established clinical
  risk factors, giving the model a defensible clinical grounding rather than
  relying on spurious correlations.

### Risk stratification sanity check

Actual positive rate increases monotonically across predicted tiers in both
datasets, confirming the tiers are clinically meaningful, not arbitrary cutoffs:

| Dataset | Low tier actual rate | Medium tier | High tier |
|---|---|---|---|
| Diabetes | 0.15 | 0.42 | 0.64 |
| Heart Disease | 0.00 | 0.45 | 0.86 |

### Fairness audit

No accuracy gap larger than 0.15 was found across age bands (both datasets) or
sex (heart dataset), meaning the models don't show strong signs of systematically
underperforming for any subgroup tested — though sample sizes for some subgroups
are small, so this should be treated as a first-pass check, not a guarantee.

## Limitations & Future Work

- Datasets are small (303–768 rows) and from specific populations (Pima Indian
  women for diabetes; a single hospital cohort for heart disease) — results
  won't generalize to other populations without revalidation.
- Fairness check uses coarse age bands and only one demographic attribute per
  dataset; a production system would need a more rigorous audit (e.g. equalized
  odds, calibration-per-group) and larger, more diverse data.
- Next steps: external validation on a second cohort, calibration analysis
  (are predicted probabilities well-calibrated?), and a lightweight deep
  learning baseline (LSTM/MLP) for direct comparison against the classical
  models used here.

## Tech Stack

Python · scikit-learn · XGBoost · SHAP · imbalanced-learn (SMOTE) · Streamlit ·
pandas · matplotlib / seaborn
