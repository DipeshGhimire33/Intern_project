import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="Sepsis Prediction",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# Load trained artifacts
# =========================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/xgboost_sepsis_model.pkl")
    weights = joblib.load("models/clinical_domain_weights.pkl")

    return model, weights


model, weights = load_artifacts()


# =========================================================
# Clinical reference ranges
# =========================================================

clinical_reference_ranges = {
    # Hemodynamic
    "HR": (60, 100),
    "SBP": (90, 140),
    "MAP": (65, 100),
    "DBP": (60, 80),

    # Respiratory
    "Resp": (12, 20),
    "O2Sat": (94, 100),
    "Temp": (36.0, 37.5),
    "EtCO2": (35, 45),
    "FiO2": (0.21, 0.60),
    "pH": (7.35, 7.45),
    "PaCO2": (35, 45),
    "SaO2": (94, 100),

    # Acid-base / metabolic
    "BaseExcess": (-2, 2),
    "HCO3": (22, 26),
    "Lactate": (0.5, 2.0),
    "Glucose": (70, 140),

    # Renal / electrolytes
    "BUN": (7, 20),
    "Creatinine": (0.6, 1.3),
    "Calcium": (8.5, 10.5),
    "Chloride": (98, 107),
    "Magnesium": (1.7, 2.2),
    "Phosphate": (2.5, 4.5),
    "Potassium": (3.5, 5.0),

    # Hepatic
    "AST": (10, 40),
    "Alkalinephos": (44, 147),
    "Bilirubin_total": (0.1, 1.2),
    "Bilirubin_direct": (0.0, 0.3),

    # Hematologic / coagulation
    "WBC": (4, 11),
    "Platelets": (150, 450),
    "Hgb": (12, 17),
    "Hct": (36, 50),
    "PTT": (25, 35),
    "Fibrinogen": (200, 400),

    # Cardiac marker
    "TroponinI": (0, 0.04),
}


# =========================================================
# Clinical domains
# =========================================================

clinical_domains = {
    "Hemodynamic": [
        "HR",
        "SBP",
        "MAP",
        "DBP"
    ],

    "Respiratory": [
        "Resp",
        "O2Sat",
        "Temp",
        "EtCO2",
        "FiO2",
        "pH",
        "PaCO2",
        "SaO2"
    ],

    "Renal_Metabolic": [
        "BaseExcess",
        "HCO3",
        "BUN",
        "Creatinine",
        "Calcium",
        "Chloride",
        "Magnesium",
        "Phosphate",
        "Potassium",
        "Glucose",
        "Lactate"
    ],

    "Inflammatory_Hematological": [
        "WBC",
        "Platelets",
        "Hgb",
        "Hct",
        "Fibrinogen"
    ],

    "Hepatic_Coagulation": [
        "AST",
        "Alkalinephos",
        "Bilirubin_total",
        "Bilirubin_direct",
        "PTT",
        "TroponinI"
    ],
}


# =========================================================
# Feature engineering functions
# =========================================================

def calculate_deviation(value, lower, upper):
    """
    Calculate normalized deviation from a reference interval.
    """

    if pd.isna(value):
        return np.nan

    reference_width = upper - lower

    if lower <= value <= upper:
        return 0.0

    elif value < lower:
        return (lower - value) / reference_width

    else:
        return (value - upper) / reference_width


def calculate_domain_score(values, weights_df, features):
    """
    Calculate a weighted domain score using the same
    missing-value handling as the preprocessing notebook.
    """

    weighted_sum = 0.0
    available_weight = 0.0

    for feature in features:

        value = values.get(feature)

        if pd.isna(value):
            continue

        feature_weight = weights_df.loc[
            feature,
            "DomainWeight"
        ]

        weighted_sum += value * feature_weight
        available_weight += feature_weight

    if available_weight == 0:
        return np.nan

    return weighted_sum / available_weight


def create_model_features(clinical_values, temporal_values):
    """
    Reproduce the project's feature-engineering pipeline
    for one patient observation.
    """

    # -----------------------------------------------------
    # Calculate clinical deviations
    # -----------------------------------------------------

    deviations = {}

    for feature, value in clinical_values.items():

        lower, upper = clinical_reference_ranges[feature]

        deviations[feature] = calculate_deviation(
            value,
            lower,
            upper
        )

    # -----------------------------------------------------
    # Calculate five clinical domain scores
    # -----------------------------------------------------

    domain_scores = {}

    for domain, features in clinical_domains.items():

        domain_scores[domain] = calculate_domain_score(
            deviations,
            weights,
            features
        )

    # -----------------------------------------------------
    # Add temporal / demographic / ICU features
    # -----------------------------------------------------

    model_features = {
        "Hemodynamic": domain_scores["Hemodynamic"],
        "Respiratory": domain_scores["Respiratory"],
        "Renal_Metabolic": domain_scores["Renal_Metabolic"],
        "Inflammatory_Hematological":
            domain_scores["Inflammatory_Hematological"],
        "Hepatic_Coagulation":
            domain_scores["Hepatic_Coagulation"],
        "Hour": temporal_values["Hour"],
        "ICULOS": temporal_values["ICULOS"],
        "HospAdmTime": temporal_values["HospAdmTime"],
        "Gender": temporal_values["Gender"],
        "ICU_Unit": temporal_values["ICU_Unit"],
    }

    return pd.DataFrame(
        [model_features],
        columns=[
            "Hemodynamic",
            "Respiratory",
            "Renal_Metabolic",
            "Inflammatory_Hematological",
            "Hepatic_Coagulation",
            "Hour",
            "ICULOS",
            "HospAdmTime",
            "Gender",
            "ICU_Unit",
        ]
    )


# =========================================================
# App header
# =========================================================

st.title("🩺 Sepsis Prediction Using Machine Learning")

st.markdown(
    """
This application uses the trained **XGBoost model** developed in the
Sepsis Prediction Using Machine Learning project.

The application accepts clinical measurements, reproduces the project's
clinical feature-engineering process, generates the five domain-level
features, and then produces an XGBoost sepsis-risk prediction.
"""
)

st.warning(
    "Educational/Research Use Only — This application is not a medical "
    "diagnostic tool and should not be used for clinical decision-making."
)


# =========================================================
# Model information
# =========================================================

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "XGBoost")

with col2:
    st.metric("Decision Threshold", "0.80")

with col3:
    st.metric("Test PR-AUC", "0.092856")


# =========================================================
# Clinical inputs
# =========================================================

st.divider()

st.header("🔬 Clinical Measurements")

st.write(
    "Enter the available clinical measurements below. "
    "Missing measurements can be left blank."
)


# =========================================================
# Hemodynamic
# =========================================================

with st.expander("❤️ Hemodynamic", expanded=True):

    col1, col2 = st.columns(2)

    with col1:

        hr = st.number_input(
            "Heart Rate (HR)",
            min_value=0.0,
            value=None,
            placeholder="e.g. 80"
        )

        sbp = st.number_input(
            "Systolic Blood Pressure (SBP)",
            min_value=0.0,
            value=None,
            placeholder="e.g. 120"
        )

    with col2:

        map_value = st.number_input(
            "Mean Arterial Pressure (MAP)",
            min_value=0.0,
            value=None,
            placeholder="e.g. 80"
        )

        dbp = st.number_input(
            "Diastolic Blood Pressure (DBP)",
            min_value=0.0,
            value=None,
            placeholder="e.g. 70"
        )


# =========================================================
# Respiratory
# =========================================================

with st.expander("🫁 Respiratory"):

    col1, col2 = st.columns(2)

    with col1:

        resp = st.number_input(
            "Respiratory Rate (Resp)",
            min_value=0.0,
            value=None,
            placeholder="e.g. 16"
        )

        o2sat = st.number_input(
            "Oxygen Saturation (O2Sat)",
            min_value=0.0,
            max_value=100.0,
            value=None,
            placeholder="e.g. 98"
        )

        temp = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            value=None,
            placeholder="e.g. 37.0"
        )

        etc02 = st.number_input(
            "End-Tidal CO2 (EtCO2)",
            min_value=0.0,
            value=None,
            placeholder="e.g. 40"
        )

    with col2:

        fio2 = st.number_input(
            "FiO2",
            min_value=0.0,
            max_value=1.0,
            value=None,
            placeholder="e.g. 0.21"
        )

        ph = st.number_input(
            "pH",
            min_value=0.0,
            max_value=14.0,
            value=None,
            placeholder="e.g. 7.40"
        )

        paco2 = st.number_input(
            "PaCO2",
            min_value=0.0,
            value=None,
            placeholder="e.g. 40"
        )

        sao2 = st.number_input(
            "SaO2",
            min_value=0.0,
            max_value=100.0,
            value=None,
            placeholder="e.g. 98"
        )


# =========================================================
# Renal / Metabolic
# =========================================================

with st.expander("🧪 Renal / Metabolic"):

    col1, col2 = st.columns(2)

    with col1:

        base_excess = st.number_input(
            "Base Excess",
            value=None,
            placeholder="e.g. 0"
        )

        hco3 = st.number_input(
            "HCO3",
            min_value=0.0,
            value=None,
            placeholder="e.g. 24"
        )

        bun = st.number_input(
            "BUN",
            min_value=0.0,
            value=None,
            placeholder="e.g. 15"
        )

        creatinine = st.number_input(
            "Creatinine",
            min_value=0.0,
            value=None,
            placeholder="e.g. 1.0"
        )

        calcium = st.number_input(
            "Calcium",
            min_value=0.0,
            value=None,
            placeholder="e.g. 9.5"
        )

        chloride = st.number_input(
            "Chloride",
            min_value=0.0,
            value=None,
            placeholder="e.g. 102"
        )

    with col2:

        magnesium = st.number_input(
            "Magnesium",
            min_value=0.0,
            value=None,
            placeholder="e.g. 2.0"
        )

        phosphate = st.number_input(
            "Phosphate",
            min_value=0.0,
            value=None,
            placeholder="e.g. 3.5"
        )

        potassium = st.number_input(
            "Potassium",
            min_value=0.0,
            value=None,
            placeholder="e.g. 4.0"
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0.0,
            value=None,
            placeholder="e.g. 100"
        )

        lactate = st.number_input(
            "Lactate",
            min_value=0.0,
            value=None,
            placeholder="e.g. 1.0"
        )


# =========================================================
# Inflammatory / Hematological
# =========================================================

with st.expander("🩸 Inflammatory / Hematological"):

    col1, col2 = st.columns(2)

    with col1:

        wbc = st.number_input(
            "White Blood Cell Count (WBC)",
            min_value=0.0,
            value=None,
            placeholder="e.g. 7"
        )

        platelets = st.number_input(
            "Platelets",
            min_value=0.0,
            value=None,
            placeholder="e.g. 250"
        )

        hgb = st.number_input(
            "Hemoglobin (Hgb)",
            min_value=0.0,
            value=None,
            placeholder="e.g. 14"
        )

    with col2:

        hct = st.number_input(
            "Hematocrit (Hct)",
            min_value=0.0,
            value=None,
            placeholder="e.g. 42"
        )

        fibrinogen = st.number_input(
            "Fibrinogen",
            min_value=0.0,
            value=None,
            placeholder="e.g. 300"
        )


# =========================================================
# Hepatic / Coagulation
# =========================================================

with st.expander("🧬 Hepatic / Coagulation / Organ Injury"):

    col1, col2 = st.columns(2)

    with col1:

        ast = st.number_input(
            "AST",
            min_value=0.0,
            value=None,
            placeholder="e.g. 25"
        )

        alkalinephos = st.number_input(
            "Alkaline Phosphatase",
            min_value=0.0,
            value=None,
            placeholder="e.g. 80"
        )

        bilirubin_total = st.number_input(
            "Total Bilirubin",
            min_value=0.0,
            value=None,
            placeholder="e.g. 0.8"
        )

    with col2:

        bilirubin_direct = st.number_input(
            "Direct Bilirubin",
            min_value=0.0,
            value=None,
            placeholder="e.g. 0.2"
        )

        ptt = st.number_input(
            "PTT",
            min_value=0.0,
            value=None,
            placeholder="e.g. 30"
        )

        troponin = st.number_input(
            "Troponin I",
            min_value=0.0,
            value=None,
            placeholder="e.g. 0.01"
        )


# =========================================================
# Temporal / demographic / ICU inputs
# =========================================================

st.divider()

st.header("🕒 Temporal & Patient Information")

col1, col2 = st.columns(2)

with col1:

    hour = st.number_input(
        "Hour",
        min_value=0,
        max_value=335,
        value=0,
        step=1
    )

    iculos = st.number_input(
        "ICULOS",
        min_value=1,
        max_value=336,
        value=1,
        step=1
    )

    hosp_adm_time = st.number_input(
        "Hospital Admission Time",
        value=0.0,
        step=1.0
    )

with col2:

    gender = st.selectbox(
        "Gender",
        options=[0, 1],
        format_func=lambda x:
            "Female (0)" if x == 0 else "Male (1)"
    )

    icu_unit = st.selectbox(
        "ICU Unit",
        options=[-1, 0, 1],
        format_func=lambda x:
            "Unknown (-1)" if x == -1 else f"Unit {x}"
    )


# =========================================================
# Build clinical input dictionary
# =========================================================

clinical_values = {
    "HR": hr,
    "SBP": sbp,
    "MAP": map_value,
    "DBP": dbp,

    "Resp": resp,
    "O2Sat": o2sat,
    "Temp": temp,
    "EtCO2": etc02,
    "FiO2": fio2,
    "pH": ph,
    "PaCO2": paco2,
    "SaO2": sao2,

    "BaseExcess": base_excess,
    "HCO3": hco3,
    "BUN": bun,
    "Creatinine": creatinine,
    "Calcium": calcium,
    "Chloride": chloride,
    "Magnesium": magnesium,
    "Phosphate": phosphate,
    "Potassium": potassium,
    "Glucose": glucose,
    "Lactate": lactate,

    "WBC": wbc,
    "Platelets": platelets,
    "Hgb": hgb,
    "Hct": hct,
    "Fibrinogen": fibrinogen,

    "AST": ast,
    "Alkalinephos": alkalinephos,
    "Bilirubin_total": bilirubin_total,
    "Bilirubin_direct": bilirubin_direct,
    "PTT": ptt,
    "TroponinI": troponin,
}


temporal_values = {
    "Hour": hour,
    "ICULOS": iculos,
    "HospAdmTime": hosp_adm_time,
    "Gender": gender,
    "ICU_Unit": icu_unit,
}


# =========================================================
# Prediction
# =========================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Sepsis Risk",
    type="primary",
    use_container_width=True
)


if predict_button:

    model_input = create_model_features(
        clinical_values,
        temporal_values
    )

    # XGBoost can handle NaN values natively.
    probability = model.predict_proba(
        model_input,
        iteration_range=(0, 48)
    )[0, 1]

    threshold = 0.80

    prediction = int(probability >= threshold)

    # -----------------------------------------------------
    # Display engineered features
    # -----------------------------------------------------

    st.subheader("Generated Model Features")

    st.dataframe(
        model_input,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # Prediction result
    # -----------------------------------------------------

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Predicted Probability",
            f"{probability:.2%}"
        )

    with result_col2:

        if prediction == 1:
            st.error(
                "⚠️ Model Prediction: Sepsis Positive"
            )
        else:
            st.success(
                "✅ Model Prediction: Sepsis Negative"
            )

    st.caption(
        f"Classification threshold: {threshold:.2f}"
    )