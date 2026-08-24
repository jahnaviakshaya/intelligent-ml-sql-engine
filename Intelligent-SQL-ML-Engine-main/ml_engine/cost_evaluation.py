import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from per_dqn_engine import DQN
from sklearn.metrics import classification_report

# Load Data
df = pd.read_csv("data/predictive_maintenance.csv")
df = df.drop(columns=[col for col in ["UDI", "Product ID", "Failure Type"] if col in df.columns])
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=["Target"]).values
y = df["Target"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# COST FUNCTION
# =============================

def compute_cost(y_true, decisions):

    total_cost = 0

    for true, action in zip(y_true, decisions):

        if action == 0:  # Do nothing
            if true == 1:
                total_cost += 100

        elif action == 1:  # Preventive
            total_cost += 20

        elif action == 2:  # Replace
            total_cost += 40

    return total_cost

# =============================
# GRADIENT BOOSTING COST
# =============================

gb_model = joblib.load("ml_engine/models/gradient_boosting.pkl")

gb_preds = gb_model.predict(X_test)

# Convert classification to maintenance decision
# If predicted failure → preventive maintenance
gb_actions = [1 if p == 1 else 0 for p in gb_preds]

gb_cost = compute_cost(y_test, gb_actions)

print("Gradient Boosting Total Cost:", gb_cost)

# =============================
# PER-DQN COST
# =============================

state_size = X_test.shape[1]
action_size = 3

policy_net = DQN(state_size, action_size)
policy_net.load_state_dict(torch.load("ml_engine/models/per_dqn_model.pth"))
policy_net.eval()

dqn_actions = []

for state in X_test:
    state_tensor = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        q_values = policy_net(state_tensor)
        action = torch.argmax(q_values).item()
    dqn_actions.append(action)

dqn_cost = compute_cost(y_test, dqn_actions)

print("PER-DQN Total Cost:", dqn_cost)
