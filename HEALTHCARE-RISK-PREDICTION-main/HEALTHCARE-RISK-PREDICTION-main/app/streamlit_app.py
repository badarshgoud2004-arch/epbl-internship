"""
Healthcare Risk Prediction — interactive demo app.

Run with:  streamlit run app/streamlit_app.py

Features:
- Choose disease model (Diabetes / Heart Disease)
- Enter patient values, get risk probability + Low/Medium/High tier
- SHAP-based explanation of the individual prediction (waterfall)
- What-if simulator: adjust a feature and see risk change live
- Model comparison dashboard (all trained models side by side)
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import config

st.set_page_config(page_title="Healthcare Risk Predictor", layout="wide")

DATASET_LABELS = {"diabetes": "Diabetes Risk", "heart": "Heart Disease Risk"}

FEATURE_HELP = {
    "pregnancies": "Number of pregnancies",
    "glucose": "Plasma glucose concentration (mg/dL)",
    "blood_pressure": "Diastolic blood pressure (mm Hg)",
    "skin_thickness": "Triceps skin fold thickness (mm)",
    "insulin": "2-Hour serum insulin (mu U/ml)",
    "bmi": "Body Mass Index",
    "diabetes_pedigree": "Diabetes pedigree function (genetic risk score)",
    "age": "Age (years)",
    "sex": "0 = female, 1 = male",
    "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting ECG results (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of peak exercise ST segment (0-2)",
    "ca": "Number of major vessels colored by fluoroscopy (0-3)",
    "thal": "Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)",
}


@st.cache_resource
def load_artifacts(dataset_name):
    model = joblib.load(os.path.join(config.MODELS_DIR, f"{dataset_name}_best_model.joblib"))
    scaler = joblib.load(os.path.join(config.MODELS_DIR, f"{dataset_name}_scaler.joblib"))
    imputer = joblib.load(os.path.join(config.MODELS_DIR, f"{dataset_name}_imputer.joblib"))
    all_models = joblib.load(os.path.join(config.MODELS_DIR, f"{dataset_name}_all_models.joblib"))
    with open(os.path.join(config.MODELS_DIR, f"{dataset_name}_results.json")) as f:
        results = json.load(f)
    raw_df = pd.read_csv(os.path.join(config.PROCESSED_DIR, f"{dataset_name}_clean.csv"))
    return model, scaler, imputer, all_models, results, raw_df


def stratify(prob):
    if prob < 0.33:
        return "Low", "🟢"
    elif prob < 0.66:
        return "Medium", "🟡"
    else:
        return "High", "🔴"


def get_shap_explainer(model, X_background):
    model_type = type(model).__name__
    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        return shap.TreeExplainer(model), "tree"
    else:
        bg = shap.sample(X_background, 50, random_state=42)
        return shap.KernelExplainer(model.predict_proba, bg), "kernel"


def main():
    st.title("🏥 Healthcare Risk Prediction System")
    st.caption("Predicts disease risk from patient data and explains *why*, using SHAP.")

    dataset_name = st.sidebar.selectbox(
        "Select model", options=list(DATASET_LABELS.keys()),
        format_func=lambda x: DATASET_LABELS[x]
    )
    model, scaler, imputer, all_models, results, raw_df = load_artifacts(dataset_name)
    feature_names = results["feature_names"]
    best_model_name = results["best_model"]

    tab1, tab2, tab3 = st.tabs(["🩺 Patient Risk Prediction", "🔀 What-If Simulator", "📊 Model Comparison"])

    # ---------------- TAB 1: Prediction ----------------
    with tab1:
        st.subheader("Enter Patient Data")
        cols = st.columns(3)
        inputs = {}
        for i, feat in enumerate(feature_names):
            col = cols[i % 3]
            default = float(raw_df[feat].median())
            lo, hi = float(raw_df[feat].min()), float(raw_df[feat].max())
            inputs[feat] = col.number_input(
                f"{feat}", value=round(default, 1), min_value=float(min(lo, 0)),
                max_value=float(hi * 1.5 if hi > 0 else hi + 10),
                help=FEATURE_HELP.get(feat, ""), key=f"pred_{feat}"
            )

        if st.button("Predict Risk", type="primary"):
            X_input = pd.DataFrame([inputs])[feature_names]
            X_imp = pd.DataFrame(imputer.transform(X_input), columns=feature_names)
            X_scaled = pd.DataFrame(scaler.transform(X_imp), columns=feature_names)

            prob = model.predict_proba(X_scaled)[0, 1]
            tier, emoji = stratify(prob)

            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Predicted Risk Probability", f"{prob:.1%}")
                st.markdown(f"### Risk Tier: {emoji} **{tier}**")

            with c2:
                st.markdown("**Why this prediction? (SHAP explanation)**")
                explainer, etype = get_shap_explainer(model, X_scaled)
                if etype == "tree":
                    sv = explainer.shap_values(X_scaled)
                    if isinstance(sv, list):
                        sv = sv[1]
                    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                        sv = sv[:, :, 1]
                    ev = explainer.expected_value
                    ev = np.atleast_1d(ev)
                    ev = ev[1] if len(ev) > 1 else ev[0]
                else:
                    sv = explainer.shap_values(X_scaled, nsamples=100)[1]
                    ev = explainer.expected_value[1]

                fig, ax = plt.subplots(figsize=(6, 4))
                exp = shap.Explanation(values=sv[0], base_values=ev,
                                        data=X_scaled.iloc[0].values, feature_names=feature_names)
                shap.plots.waterfall(exp, show=False, max_display=8)
                st.pyplot(fig, clear_figure=True)

    # ---------------- TAB 2: What-If Simulator ----------------
    with tab2:
        st.subheader("Explore how one feature affects predicted risk")
        st.caption("Fix all other patient values, then sweep one feature across its range to see risk change live.")

        sim_feature = st.selectbox("Feature to vary", feature_names, key="sim_feat")

        cols = st.columns(3)
        base_inputs = {}
        for i, feat in enumerate(feature_names):
            if feat == sim_feature:
                continue
            col = cols[i % 3]
            default = float(raw_df[feat].median())
            base_inputs[feat] = col.number_input(
                f"{feat}", value=round(default, 1), help=FEATURE_HELP.get(feat, ""), key=f"sim_{feat}"
            )

        lo, hi = float(raw_df[sim_feature].min()), float(raw_df[sim_feature].max())
        sweep_values = np.linspace(lo, hi, 25)

        rows = []
        for v in sweep_values:
            row = dict(base_inputs)
            row[sim_feature] = v
            rows.append(row)
        sweep_df = pd.DataFrame(rows)[feature_names]
        sweep_imp = pd.DataFrame(imputer.transform(sweep_df), columns=feature_names)
        sweep_scaled = pd.DataFrame(scaler.transform(sweep_imp), columns=feature_names)
        sweep_probs = model.predict_proba(sweep_scaled)[:, 1]

        chart_df = pd.DataFrame({sim_feature: sweep_values, "predicted_risk": sweep_probs})
        st.line_chart(chart_df.set_index(sim_feature))
        st.caption(f"As **{sim_feature}** changes (all else held fixed), predicted risk moves as shown above. "
                   f"Useful for understanding the model's sensitivity to a single risk factor.")

    # ---------------- TAB 3: Model Comparison ----------------
    with tab3:
        st.subheader(f"All models trained on {DATASET_LABELS[dataset_name]} dataset")
        metrics_df = pd.DataFrame({
            m: {k: v for k, v in r.items() if k != "best_params"}
            for m, r in results["results"].items()
        }).T
        st.dataframe(metrics_df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)
        st.markdown(f"**Best model selected (by ROC-AUC): `{best_model_name}`**")

        st.bar_chart(metrics_df[["accuracy", "precision", "recall", "f1", "roc_auc"]])

        fairness_path = os.path.join(config.MODELS_DIR, f"{dataset_name}_fairness_report.json")
        if os.path.exists(fairness_path):
            with open(fairness_path) as f:
                fairness = json.load(f)
            st.markdown("**Fairness check (performance across subgroups):**")
            for group_type, subgroups in fairness.items():
                st.write(f"_{group_type}_")
                st.dataframe(pd.DataFrame(subgroups).T, use_container_width=True)


if __name__ == "__main__":
    main()
