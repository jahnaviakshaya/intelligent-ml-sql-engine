import sqlite3
import os
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 🚀 FEATURE 3: Deep Learning UDF
# ------------------------------------------------------------
from ml_engine.dl_model import dl_predict_logic

# 🚀 FEATURE 6: PER-DQN
from ml_engine.per_dqn_engine import DQN

# ------------------------------------------------------------
# 🚀 PERFORMANCE: SQL MODEL CACHE (NEW)
# ------------------------------------------------------------
_sql_model_cache = None

# ============================================================
# FEATURE 1: Query Complexity Estimator
# ============================================================
def estimate_query_complexity(sql):
    score = 0
    sql_upper = sql.upper()
    score += sql_upper.count("CASE") + sql_upper.count("WHEN")
    score += sql_upper.count("JOIN") * 5
    score += sql_upper.count("GROUP BY") * 3
    score += sql_upper.count("DL_PREDICT") * 4
    return score

# ============================================================
# FEATURE 3: Register DL UDF
# ============================================================
def register_dl_udf(conn):
    conn.create_function("DL_PREDICT", 1, dl_predict_logic)

# ============================================================
# FEATURE 4: Feedback Monitor
# ============================================================
def should_retrain():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, '..', 'db', 'intelligent_db.sqlite')

        conn = sqlite3.connect(db_path)
        df = pd.read_sql(
            "SELECT AVG(correct) AS acc, COUNT(*) AS cnt FROM model_feedback",
            conn
        )
        conn.close()

        return (
            df.iloc[0]['cnt'] >= 5 and
            df.iloc[0]['acc'] is not None and
            df.iloc[0]['acc'] < 0.85
        )
    except Exception:
        return False

# ============================================================
# FEATURE 5: Materialized ML View
# ============================================================
def write_back_results(df, table_name="predicted_results"):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, '..', 'db', 'intelligent_db.sqlite')
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        return True
    except Exception as e:
        print(f"Write-back error: {e}")
        return False

# ============================================================
# STEP 1: SQL Decision Tree Injection (OPTIMIZED)
# ============================================================
def inject_sql_model():
    global _sql_model_cache

    if _sql_model_cache is not None:
        return _sql_model_cache

    base_dir = os.path.dirname(os.path.abspath(__file__))
    sql_path = os.path.join(base_dir, '..', 'ml_engine', 'model_logic.sql')

    try:
        with open(sql_path, "r") as f:
            _sql_model_cache = f.read()
            return _sql_model_cache
    except FileNotFoundError:
        return "NULL /* ERROR: model_logic.sql not found */"

# ============================================================
# 🚀 INTELLIGENT QUERY PROCESSOR
# ============================================================
def intelligent_query_processor(user_query, user_role="Staff"):

    explain_requested = "WITH EXPLANATION" in user_query.upper()
    clean_query = user_query.replace("WITH EXPLANATION", "").strip()

    # RBAC Guard
    if user_role != "Admin" and "EXPLAINABLE_AI_REASONING" in clean_query.upper():
        return None, {"error": "Access Denied."}

    # Cost-Based Optimizer
    complexity = estimate_query_complexity(clean_query)
    cost = (1500 * 0.5) + (complexity * 10)

    strategy = (
        "Hybrid UDF Execution (DL)"
        if "DL_PREDICT" in clean_query.upper()
        else "Real-Time In-DB Execution"
        if cost < 800
        else "Vectorized Batch Execution"
    )

    rewritten_query = clean_query

    # Inject Decision Tree SQL
    if "PREDICT(failure)" in rewritten_query:
        dynamic_model_sql = inject_sql_model()
        rewritten_query = rewritten_query.replace(
            "PREDICT(failure)",
            f"(\n{dynamic_model_sql}\n) AS Predicted_Failure"
        )

    retrain_alert = should_retrain()

    optimizer_info = {
        "strategy": strategy,
        "estimated_cost": cost,
        "query_complexity": complexity,
        "user_role": user_role,
        "model_status": (
            "🚨 RETRAINING TRIGGERED"
            if retrain_alert
            else "✅ MODEL STABLE"
        )
    }

    return rewritten_query.strip(), optimizer_info

# ============================================================
# STEP 2: PER-DQN Maintenance Optimizer (OPTIMIZED)
# ============================================================
def optimize_maintenance():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, '..', 'db', 'intelligent_db.sqlite')
    model_path = os.path.join(base_dir, '..', 'ml_engine', 'models', 'per_dqn_model.pth')
    scaler_path = os.path.join(base_dir, '..', 'ml_engine', 'models', 'per_dqn_scaler.pkl')

    if not os.path.exists(model_path):
        print("⚠ PER-DQN model not found.")
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM machines", conn)
    conn.close()

    if df.empty:
        return pd.DataFrame()

    cols_to_drop = [col for col in ["UDI", "Product ID", "Failure Type", "Target"] if col in df.columns]
    X_df = df.drop(columns=cols_to_drop)
    X_df = pd.get_dummies(X_df, drop_first=True)

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X_df.values)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df.values)

    state_size = X_scaled.shape[1]
    action_size = 3

    model = DQN(state_size, action_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 🚀 PERFORMANCE FIX: BATCH INFERENCE (NO LOGIC CHANGE)
    states = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        q_values = model(states)
        actions = torch.argmax(q_values, dim=1).numpy()

    results = actions.tolist()

    df["recommended_action"] = results

    action_map = {
        0: "Do Nothing",
        1: "Preventive Maintenance",
        2: "Replace Component"
    }

    df["action_label"] = df["recommended_action"].map(action_map)

    return df

# ============================================================
# STEP 3: Business Cost Computation
# ============================================================
def compute_cost(y_true, actions):

    total_cost = 0

    for true, action in zip(y_true, actions):

        if action == 0:
            if true == 1:
                total_cost += 100
        elif action == 1:
            total_cost += 20
        elif action == 2:
            total_cost += 40

    return total_cost

# ============================================================
# STEP 4: SQL Execution Engine
# ============================================================
def execute_sql(query):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, '..', 'db', 'intelligent_db.sqlite')

    conn = sqlite3.connect(db_path)
    register_dl_udf(conn)

    cursor = conn.cursor()

    try:
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        results = cursor.fetchall()
        return columns, results
    except Exception as e:
        return [], f"Database Execution Error: {str(e)}"
    finally:
        conn.close()

def process_and_execute_query(user_query, user_role="Staff"):

    rewritten_query, optimizer_info = intelligent_query_processor(user_query, user_role)

    if rewritten_query is None:
        return {"error": optimizer_info["error"]}

    columns, results = execute_sql(rewritten_query)

    return {
        "rewritten_query": rewritten_query,
        "optimizer_info": optimizer_info,
        "columns": columns,
        "results": results
    }