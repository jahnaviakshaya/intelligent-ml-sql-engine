import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# ==============================
# 1. LOAD DATA (NO SCALING)
# ==============================
# Note: We DO NOT use StandardScaler here.
# The Decision Tree must learn thresholds based on the RAW data
# so the resulting SQL query matches the live values in your SQLite DB.

DATA_PATH = "data/predictive_maintenance.csv"
df = pd.read_csv(DATA_PATH)

cols_to_drop = [col for col in ["UDI", "Product ID", "Failure Type"] if col in df.columns]
df = df.drop(columns=cols_to_drop)
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=["Target"])
y = df["Target"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 2. TRAIN DECISION TREE
# ==============================
# max_depth=5 ensures the resulting SQL query is extremely fast 
# and doesn't exceed SQLite's parser limits.

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\nDecision Tree Report:")
print(classification_report(y_test, preds))

# ==============================
# 3. SAVE MODEL (FOLDER SAFE)
# ==============================

os.makedirs("ml_engine/models", exist_ok=True)
joblib.dump(model, "ml_engine/models/decision_tree.pkl")

# ==============================
# 4. CONVERT TREE TO NATIVE SQL
# ==============================

def tree_to_sql(tree_model, features):
    """
    Recursively traverses the scikit-learn Decision Tree and 
    generates a nested SQL CASE WHEN statement.
    """
    tree_ = tree_model.tree_
    
    def recurse(node, depth):
        indent = "  " * depth
        
        # -2 is the scikit-learn internal indicator for a leaf node (TREE_UNDEFINED)
        if tree_.feature[node] != -2:
            # It's a decision node
            name = features[tree_.feature[node]]
            threshold = tree_.threshold[node]
            
            # Format column name safely for SQL
            safe_name = f'"{name}"'
            
            left_query = recurse(tree_.children_left[node], depth + 1)
            right_query = recurse(tree_.children_right[node], depth + 1)
            
            return (f"{indent}CASE\n"
                    f"{indent}  WHEN {safe_name} <= {threshold:.4f} THEN\n{left_query}\n"
                    f"{indent}  ELSE\n{right_query}\n"
                    f"{indent}END")
        else:
            # It's a leaf node. Get the predicted class (0 or 1).
            class_idx = np.argmax(tree_.value[node])
            predicted_class = tree_model.classes_[class_idx]
            return f"{indent}  {predicted_class}"
            
    return recurse(0, 0)

sql_logic = tree_to_sql(model, feature_names)

# Save the raw SQL logic to a file for the Middleware to inject
with open("ml_engine/model_logic.sql", "w") as f:
    f.write(sql_logic)

print("\nModel converted to SQL successfully. Saved to ml_engine/model_logic.sql")