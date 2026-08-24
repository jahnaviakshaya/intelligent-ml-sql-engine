import shap
import joblib
import pandas as pd
import os

# ------------------------------------------------------------
# Path Configuration
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "gradient_boosting.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "gradient_boosting_scaler.pkl")

# ------------------------------------------------------------
# Model & Scaler Loader (Enhanced)
# ------------------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file missing at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_scaler():
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file missing at {SCALER_PATH}")
    return joblib.load(SCALER_PATH)


# ------------------------------------------------------------
# Internal Utility: Align + Scale Features Safely
# ------------------------------------------------------------
def _align_and_scale(input_df, model):
    """
    Ensures:
    1. Correct feature names
    2. Correct feature order
    3. Correct scaling
    """

    # ✅ Use official sklearn stored schema
    model_features = list(model.feature_names_in_)

    # Check missing features
    missing = set(model_features) - set(input_df.columns)
    if missing:
        raise ValueError(
            f"Missing required model features: {missing}. "
            f"Run a row-level query including all original feature columns."
        )

    # Remove extra columns safely
    aligned_df = input_df[model_features]

    # Apply scaler (same used during training)
    scaler = load_scaler()
    scaled = scaler.transform(aligned_df)

    # Convert back to DataFrame to preserve column names
    scaled_df = pd.DataFrame(scaled, columns=model_features)

    return scaled_df


# ------------------------------------------------------------
# SHAP & Feature Importance (Fully Safe Version)
# ------------------------------------------------------------
def generate_shap_values(input_df):
    model = load_model()

    # 🔥 Alignment + Scaling Protection
    input_df = _align_and_scale(input_df, model)

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="interventional"
    )

    shap_values = explainer.shap_values(
        input_df,
        check_additivity=False
    )

    return shap_values, explainer


def global_feature_importance(input_df):
    model = load_model()

    # Align & scale first
    input_df = _align_and_scale(input_df, model)

    shap_values, _ = generate_shap_values(input_df)

    importance = pd.DataFrame({
        "feature": model.feature_names_in_,
        "importance": abs(shap_values).mean(axis=0)
    }).sort_values(by="importance", ascending=False)

    return importance


# ------------------------------------------------------------
# Prediction Probability (Fully Schema-Safe)
# ------------------------------------------------------------
def get_prediction_probability(X_row):
    model = load_model()

    # Convert Series to DataFrame
    if isinstance(X_row, pd.Series):
        X_row = X_row.to_frame().T

    X_row = _align_and_scale(X_row, model)

    return model.predict_proba(X_row)[0][1]


# ------------------------------------------------------------
# Human-Readable Explanation (Preserved + Stable)
# ------------------------------------------------------------
def generate_human_explanation(row, prob):

    torque = row.get("Torque [Nm]", "N/A")
    wear = row.get("Tool wear [min]", "N/A")
    air_temp = row.get("Air temperature [K]", "N/A")
    process_temp = row.get("Process temperature [K]", "N/A")

    if prob < 0.3:
        return (
            f"✅ **LOW RISK ({prob:.1%}):** The machine is operating under safe conditions. "
            f"Torque ({torque} Nm) and tool wear ({wear} min) are within acceptable limits. "
            f"No immediate maintenance action is required."
        )

    elif prob < 0.7:
        return (
            f"⚠️ **MODERATE RISK ({prob:.1%}):** Mechanical stress indicators are increasing. "
            f"Torque ({torque} Nm) or tool wear ({wear} min) appear elevated. "
            f"Air temperature ({air_temp}K) and process temperature ({process_temp}K) "
            f"should be monitored closely."
        )

    else:
        return (
            f"🚨 **HIGH RISK ({prob:.1%}):** Immediate intervention recommended. "
            f"Extreme torque ({torque} Nm) and excessive tool wear ({wear} min) detected. "
            f"Thermal stress (Air: {air_temp}K, Process: {process_temp}K) "
            f"indicates high probability of imminent breakdown."
        )