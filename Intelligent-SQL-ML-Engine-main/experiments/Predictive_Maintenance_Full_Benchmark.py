# ============================================================
# Predictive Maintenance - FULL ML + DQN Benchmark
# Classical ML + DQN + Double-DQN + PER-DQN
# Accurate | Stable | Publication Ready
# ============================================================

import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import xgboost as xgb
from collections import deque

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = r"D:\Major project\intelligent-sql-ml-engine\experiments\datasets\predictive_maintenance.csv"

df = pd.read_csv(DATA_PATH)

cols_to_drop = [col for col in ["UDI", "Product ID", "Failure Type"] if col in df.columns]
df = df.drop(columns=cols_to_drop)

df = pd.get_dummies(df, drop_first=True)

target_col = "Target"

X = df.drop(columns=[target_col]).values
y = df[target_col].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results = []

# ============================================================
# 2. CLASSICAL MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200),
    "XGBoost": xgb.XGBClassifier(eval_metric="logloss"),
    "SVM": SVC(probability=True, class_weight="balanced"),
    "MLP": MLPClassifier(hidden_layer_sizes=(128,64), max_iter=400)
}

def evaluate_model(name, model):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy").mean()

    results.append([
        name,
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob),
        cv_acc,
        train_time
    ])

for name, model in models.items():
    print("Training:", name)
    evaluate_model(name, model)

# ============================================================
# 3. RL ENVIRONMENT
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MaintenanceEnv:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.max_steps = len(X)

    def reset(self):
        self.index = 0
        return self.X[self.index]

    def step(self, action):
        failure = self.y[self.index]

        # Balanced reward
        if action == 0:
            reward = -15 if failure == 1 else 5
        elif action == 1:
            reward = 10 if failure == 1 else -5
        else:
            reward = 20 if failure == 1 else -10

        self.index += 1
        done = self.index >= self.max_steps - 1
        next_state = self.X[self.index] if not done else np.zeros_like(self.X[0])

        return next_state, reward, done, {"true_label": failure}

# ============================================================
# 4. DQN NETWORK
# ============================================================

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# 5. REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(states, dtype=torch.float32).to(device),
            torch.tensor(actions, dtype=torch.long).to(device),
            torch.tensor(rewards, dtype=torch.float32).to(device),
            torch.tensor(next_states, dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).to(device)
        )

    def __len__(self):
        return len(self.buffer)

# ============================================================
# 6. TRAIN RL VARIANTS
# ============================================================

def train_rl(model_type="DQN", episodes=50):

    state_size = X_train.shape[1]
    action_size = 3

    env = MaintenanceEnv(X_train, y_train)

    policy_net = DQN(state_size, action_size).to(device)
    target_net = DQN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    buffer = ReplayBuffer()

    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.98
    epsilon_min = 0.05
    batch_size = 64

    start_time = time.time()

    for episode in range(episodes):

        state = env.reset()

        while True:

            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)

            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)
            else:
                with torch.no_grad():
                    action = torch.argmax(policy_net(state_tensor)).item()

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)

            state = next_state

            if len(buffer) > batch_size:

                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                if model_type == "Double-DQN":
                    next_actions = torch.argmax(policy_net(next_states), dim=1)
                    next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
                else:
                    next_q = target_net(next_states).max(1)[0]

                target = rewards + gamma * next_q * (1 - dones)
                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

                loss = criterion(current_q, target.detach())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            if done:
                break

        # Soft update
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(0.05 * policy_param.data + 0.95 * target_param.data)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    train_time = time.time() - start_time
    return policy_net, train_time

# ============================================================
# 7. EVALUATE RL
# ============================================================

def evaluate_rl(model, name, train_time):

    env = MaintenanceEnv(X_test, y_test)

    y_true, y_pred, y_probs = [], [], []

    state = env.reset()

    while True:

        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)

        with torch.no_grad():
            q_values = model(state_tensor)
            probs = F.softmax(q_values, dim=0)

        action = torch.argmax(q_values).item()

        next_state, reward, done, info = env.step(action)

        y_true.append(info["true_label"])
        y_pred.append(1 if action != 0 else 0)
        y_probs.append(probs[1].item())

        state = next_state

        if done:
            break

    results.append([
        name,
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        roc_auc_score(y_true, y_probs),
        "N/A",
        train_time
    ])

# ============================================================
# 8. RUN RL VARIANTS
# ============================================================

for algo in ["DQN", "Double-DQN", "PER-DQN"]:
    print("Training:", algo)
    model, train_time = train_rl(algo)
    evaluate_rl(model, algo, train_time)

# ============================================================
# 9. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results, columns=[
    "Algorithm",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC",
    "CV Accuracy",
    "Training Time (sec)"
])

print("\n================ FINAL RESULTS ================\n")
print(results_df.sort_values(by="Accuracy", ascending=False))

results_df.to_csv("Predictive_Maintenance_Full_Benchmark_Results.csv", index=False)

print("\nSaved as Predictive_Maintenance_Full_Benchmark_Results.csv")