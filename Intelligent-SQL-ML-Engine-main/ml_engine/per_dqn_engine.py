# =========================================================
# DOUBLE DQN WITH PRIORITIZED EXPERIENCE REPLAY (PER)
# Predictive Maintenance - Intelligent SQL ML Engine
# =========================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os

# =========================================================
# 1️⃣ LOAD DATA
# =========================================================

DATA_PATH = r"data/predictive_maintenance.csv"

df = pd.read_csv(DATA_PATH)

# Drop ID & leakage columns safely
cols_to_drop = [col for col in ["UDI", "Product ID", "Failure Type"] if col in df.columns]
df = df.drop(columns=cols_to_drop)

# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=["Target"]).values
y = df["Target"].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Stratified split (important for imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# 2️⃣ ENVIRONMENT
# =========================================================

class MaintenanceEnv:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.max_steps = len(X)
        self.reset()

    def reset(self):
        self.index = 0
        return self.X[self.index]

    def step(self, action):

        failure = self.y[self.index]

        # Cost-aware reward shaping
        if action == 0:  # Do nothing
            reward = -80 if failure == 1 else 5
        elif action == 1:  # Preventive
            reward = 25 if failure == 1 else -8
        elif action == 2:  # Replace
            reward = 40 if failure == 1 else -15

        # Normalize reward (stabilizes training)
        reward = reward / 100.0

        self.index += 1
        done = self.index >= self.max_steps - 1

        next_state = self.X[self.index] if not done else np.zeros_like(self.X[0])

        return next_state, reward, done, {"true_label": failure}

# =========================================================
# 3️⃣ DQN NETWORK
# =========================================================

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

# =========================================================
# 4️⃣ PRIORITIZED EXPERIENCE REPLAY
# =========================================================

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):

        max_priority = max(self.priorities, default=1.0)

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):

        priorities = np.array(self.priorities) ** self.alpha
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[i] for i in indices]

        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
            indices
        )

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error.item()) + 1e-5

    def __len__(self):
        return len(self.buffer)

# =========================================================
# 5️⃣ TRAIN PER-DOUBLE-DQN
# =========================================================

def train_per_dqn(episodes=150):

    env = MaintenanceEnv(X_train, y_train)

    state_size = X_train.shape[1]
    action_size = 3

    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    buffer = PrioritizedReplayBuffer()

    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.997
    epsilon_min = 0.01
    batch_size = 64

    for episode in range(episodes):

        state = env.reset()
        total_reward = 0

        while True:

            state_tensor = torch.tensor(state, dtype=torch.float32)

            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)

            buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if len(buffer) > batch_size:

                states, actions, rewards, next_states, dones, indices = buffer.sample(batch_size)

                # Double DQN target calculation
                next_actions = torch.argmax(policy_net(next_states), dim=1)
                next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()

                target = rewards + gamma * next_q * (1 - dones)
                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

                td_errors = target - current_q

                loss = criterion(current_q, target.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                buffer.update_priorities(indices, td_errors.detach())

            if done:
                break

        target_net.load_state_dict(policy_net.state_dict())
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {episode+1}, Reward: {total_reward}")

    # =====================================================
    # SAVE MODEL + SCALER (CRITICAL FOR MIDDLEWARE)
    # =====================================================

    os.makedirs("ml_engine/models", exist_ok=True)

    torch.save(policy_net.state_dict(), "ml_engine/models/per_dqn_model.pth")
    joblib.dump(scaler, "ml_engine/models/per_dqn_scaler.pkl")

    print("\nPER-DQN Training Complete.")
    return policy_net

# =========================================================
# 6️⃣ EVALUATION (Classification Metrics)
# =========================================================

def evaluate(model):

    env = MaintenanceEnv(X_test, y_test)

    y_true = []
    y_pred = []
    y_probs = []

    state = env.reset()

    while True:
        state_tensor = torch.tensor(state, dtype=torch.float32)

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

    print("\nPER-DQN Classification Metrics:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_probs))
    
    # ✅ Added Confusion Matrix Output Here
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # ==============================
    # SAVE METRICS ARTIFACT (UPDATED FORMAT)
    # ==============================
    os.makedirs("ml_engine/reports", exist_ok=True)
    
    metrics = {
        "Model": "PER-DQN",
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_probs)
    }

    pd.DataFrame([metrics]).to_csv(
        "ml_engine/reports/per_dqn_metrics.csv", 
        index=False
    )
    print("✅ Metrics saved to 'ml_engine/reports/per_dqn_metrics.csv'")

# =========================================================
# 7️⃣ MAIN
# =========================================================

if __name__ == "__main__":

    model = train_per_dqn()
    evaluate(model)