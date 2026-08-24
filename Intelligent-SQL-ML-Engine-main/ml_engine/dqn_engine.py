# =========================================================
# ADVANCED DOUBLE DQN FOR PREDICTIVE MAINTENANCE
# Includes:
# - Double DQN
# - Reward Plotting
# - Classification Metrics
# - Clean Tensor Handling
# =========================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

# =========================================================
# 1. LOAD DATA
# =========================================================

DATA_PATH = r"D:\House_Price_Prediction\intelligent-sql-ml-engine\data\predictive_maintenance.csv"

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

# =========================================================
# 2. ENVIRONMENT
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

        if action == 0:
            reward = -100 if failure == 1 else 5
        elif action == 1:
            reward = 20 if failure == 1 else -10
        elif action == 2:
            reward = 30 if failure == 1 else -20

        self.index += 1
        done = self.index >= self.max_steps - 1
        next_state = self.X[self.index] if not done else np.zeros_like(self.X[0])

        return next_state, reward, done, {"true_label": failure}


# =========================================================
# 3. DQN NETWORK
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
# 4. REPLAY BUFFER
# =========================================================

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


# =========================================================
# 5. TRAIN DOUBLE DQN
# =========================================================

def train_dqn(episodes=100):

    state_size = X_train.shape[1]
    action_size = 3

    env = MaintenanceEnv(X_train, y_train)

    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    buffer = ReplayBuffer()

    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    batch_size = 64

    reward_history = []

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

                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                # DOUBLE DQN
                next_actions = torch.argmax(policy_net(next_states), dim=1)
                next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()

                target = rewards + gamma * next_q * (1 - dones)

                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

                loss = criterion(current_q, target.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        reward_history.append(total_reward)
        print(f"Episode {episode+1}, Reward: {total_reward}")

    os.makedirs("models", exist_ok=True)
    torch.save(policy_net.state_dict(), "models/dqn_model.pth")

    # Plot reward curve
    plt.plot(reward_history)
    plt.title("Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

    return policy_net


# =========================================================
# 6. EVALUATION WITH CLASSIFICATION METRICS
# =========================================================

def evaluate_dqn(model):

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

    print("\nDQN Classification Metrics:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_probs))


# =========================================================
# 7. MAIN
# =========================================================

if __name__ == "__main__":

    model = train_dqn(episodes=100)
    evaluate_dqn(model)
