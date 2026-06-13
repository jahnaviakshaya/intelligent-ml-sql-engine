This project enables machine learning directly inside SQL queries using a custom PREDICT(failure) operator. It eliminates data movement to external systems and enables real-time, explainable, and cost-aware analytics within the database.# intelligent-ml-sql-engine

 Problem
Traditional ML pipelines:
Move data outside databases → latency
Require separate ML infrastructure
Lack real-time decision-making

Solution
A middleware-driven system that:

Converts Decision Tree → SQL CASE logic
Executes predictions directly inside the database
Uses Reinforcement Learning (PER-DQN) for cost optimization
Provides Explainable AI (SHAP)


Key Features
SQL-based ML (PREDICT() operator)
In-database inference
Decision Tree → SQL conversion
Cost-Based Optimization (CBO)
Reinforcement Learning (PER-DQN)
Explainable AI (SHAP)
Interactive Streamlit UI

Tech Stack
Python
SQLite
scikit-learn
PyTorch
SHAP
Streamlit





## 📸 Demo

### Streamlit Application Output
![Demo](assets/demo.png)
