import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# ✅ Full metrics (UNCHANGED)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

DATA_PATH = r"data/predictive_maintenance.csv"

# ==============================
# DATA LOADING & PREPROCESSING
# ==============================
df = pd.read_csv(DATA_PATH)

# Drop unnecessary columns safely
df = df.drop(columns=[col for col in ["UDI", "Product ID", "Failure Type"] if col in df.columns])

# One-hot encoding (UNCHANGED)
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop(columns=["Target"])
y = df["Target"]

# 🔥 NEW: Preserve feature names BEFORE scaling
feature_names = X.columns

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔥 NEW: Convert back to DataFrame to preserve feature names
X = pd.DataFrame(X_scaled, columns=feature_names)

# Train-test split (UNCHANGED)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# MODEL TRAINING
# ==============================
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# ==============================
# FULL METRIC EVALUATION
# ==============================
probs = model.predict_proba(X_test)[:, 1]

# Default threshold 0.5
y_pred = (probs > 0.5).astype(int)

print("\nGradient Boosting Detailed Report:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, probs))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nFull Classification Report:")
print(classification_report(y_test, y_pred))

# ==============================
# SAVE MODEL & SCALER
# ==============================
os.makedirs("ml_engine/models", exist_ok=True)

joblib.dump(model, "ml_engine/models/gradient_boosting.pkl")
joblib.dump(scaler, "ml_engine/models/gradient_boosting_scaler.pkl")  # 🔥 NEW

print("\nModel saved successfully to 'ml_engine/models/gradient_boosting.pkl'")
print("Scaler saved successfully to 'ml_engine/models/gradient_boosting_scaler.pkl'")

# ==============================
# SAVE METRICS ARTIFACT
# ==============================
os.makedirs("ml_engine/reports", exist_ok=True)

metrics = {
    "Model": "Gradient Boosting",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "ROC_AUC": roc_auc_score(y_test, probs)
}

pd.DataFrame([metrics]).to_csv(
    "ml_engine/reports/gradient_boosting_metrics.csv",
    index=False
)

print("✅ Metrics saved to 'ml_engine/reports/gradient_boosting_metrics.csv'")