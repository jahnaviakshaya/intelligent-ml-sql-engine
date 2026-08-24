import streamlit as st
import pandas as pd
import sqlite3
import os
import sys
import plotly.express as px
import shap
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Path Resolution
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ------------------------------------------------------------
# Middleware Imports (Preserved)
# ------------------------------------------------------------
from middleware.middleware import (
    intelligent_query_processor,
    write_back_results,
    register_dl_udf,
    optimize_maintenance,
    compute_cost
)

# ------------------------------------------------------------
# XAI Imports (Enhanced)
# ------------------------------------------------------------
from ml_engine.xai import (
    generate_shap_values,
    global_feature_importance,
    get_prediction_probability,
    generate_human_explanation
)

from ml_engine.tree_explainer import explain_prediction

# ------------------------------------------------------------
# Streamlit Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="In-Database Machine Learning Engine",
    layout="wide"
)

DB_PATH = os.path.join(PROJECT_ROOT, 'db', 'intelligent_db.sqlite')

# ------------------------------------------------------------
# 🚀 PERFORMANCE: Cached DB Connection (NEW)
# ------------------------------------------------------------
@st.cache_resource
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

# ------------------------------------------------------------
# 🚀 PERFORMANCE: Cached Query Execution (NEW)
# ------------------------------------------------------------
@st.cache_data
def run_query(query):
    conn = get_connection()
    return pd.read_sql_query(query, conn)

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("🔐 Security Portal")
role = st.sidebar.selectbox("Access Level:", ["Staff", "Admin"])

st.sidebar.divider()
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "🧠 Intelligent SQL",
        "📊 Model Performance",
        "💰 Cost Optimization",
        "🔧 Maintenance Optimizer",
        "🧠 Explainable AI",
        "🏗 Architecture"
    ]
)

# ============================================================
# PAGE 1 — Intelligent SQL
# ============================================================
if page == "🧠 Intelligent SQL":

    st.title("🧠 Intelligent SQL-ML Engine")
    st.write("Hybrid Supervised + Reinforcement Learning with Cost-Based Optimization")

    # ✅ LIMIT FIX
    default_query = "SELECT *, PREDICT(failure) FROM machines LIMIT 100"

    user_query = st.text_area("SQL Input:", default_query, height=120)

    if st.button("🚀 Execute Optimized Query"):

        rewritten_sql, info = intelligent_query_processor(
            user_query,
            user_role=role
        )

        if rewritten_sql is None:
            st.error(info["error"])
        else:
            if "🚨" in info.get("model_status", ""):
                st.sidebar.warning(info["model_status"])
            else:
                st.sidebar.success(info.get("model_status", "✅ MODEL STABLE"))

            col1, col2, col3 = st.columns(3)
            col1.metric("Execution Strategy", info["strategy"])
            col2.metric("Estimated Cost", info["estimated_cost"])
            col3.metric("User Role", info["user_role"])

            if not os.path.exists(DB_PATH):
                st.error("❌ Database not found.")
            else:
                conn = get_connection()
                register_dl_udf(conn)

                try:
                    # ✅ PERFORMANCE SPINNER + CACHE
                    with st.spinner("Executing intelligent query..."):
                        df = run_query(rewritten_sql)

                    st.session_state["last_query_df"] = df.copy()

                    # ✅ OPTIONAL WRITE BACK (NO REMOVAL)
                    wb_status = False
                    if st.checkbox("💾 Save results to database"):
                        wb_status = write_back_results(df)

                    st.subheader("📊 Prediction Results")

                    if wb_status:
                        st.success("Results materialized to database.")

                    st.dataframe(df, use_container_width=True)

                    if role == "Admin":
                        with st.expander("🔍 Explainability Info"):
                            st.info("Decision Tree rules injected directly into SQL CASE statements.")

                    with st.expander("🛠️ View Rewritten SQL"):
                        st.code(rewritten_sql, language="sql")

                except Exception as e:
                    st.error(f"SQL Execution Error: {e}")

# ============================================================
# PAGE 2 — Model Performance (UNCHANGED)
# ============================================================
elif page == "📊 Model Performance":

    st.title("📊 Model Performance Dashboard")

    reports_path = os.path.join(PROJECT_ROOT, "ml_engine", "reports")
    models = []

    try:
        gb = pd.read_csv(os.path.join(reports_path, "gradient_boosting_metrics.csv"))
        models.append(gb)
    except:
        pass

    try:
        dqn = pd.read_csv(os.path.join(reports_path, "per_dqn_metrics.csv"))
        models.append(dqn)
    except:
        pass

    if len(models) > 0:
        df = pd.concat(models, ignore_index=True)
        st.dataframe(df, use_container_width=True)

        fig = px.bar(df, x="Model", y="F1", color="Model", title="F1 Score Comparison")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No model reports found.")

# ============================================================
# PAGE 3 — Cost Optimization (UNCHANGED)
# ============================================================
elif page == "💰 Cost Optimization":

    st.title("💰 Maintenance Cost Analysis")

    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM machines", conn)

    rl_df = optimize_maintenance()

    rl_cost = compute_cost(
        rl_df["Target"],
        rl_df["recommended_action"]
    ) if not rl_df.empty else None

    rewritten_sql, _ = intelligent_query_processor(
        "SELECT *, PREDICT(failure) FROM machines",
        user_role=role
    )

    sql_df = run_query(rewritten_sql)

    if "Predicted_Failure" in sql_df.columns:
        sql_actions = sql_df["Predicted_Failure"].apply(lambda x: 1 if x == 1 else 0)
        sql_cost = compute_cost(sql_df["Target"], sql_actions)
    else:
        sql_cost = None

    c1, c2 = st.columns(2)

    if rl_cost is not None:
        c1.metric("RL Optimizer Cost", rl_cost)

    if sql_cost is not None:
        c2.metric("SQL Tree Cost", sql_cost)

# ============================================================
# PAGE 4 — Maintenance Optimizer (UNCHANGED)
# ============================================================
elif page == "🔧 Maintenance Optimizer":

    st.title("🔧 RL-Based Maintenance Recommendation")

    if st.button("⚙️ Run RL Optimizer"):

        df = optimize_maintenance()

        if df.empty:
            st.error("PER-DQN model not found.")
        else:
            st.dataframe(df, use_container_width=True)

            if "Target" in df.columns:
                cost = compute_cost(df["Target"], df["recommended_action"])
                st.metric("💰 Total Business Cost (RL)", cost)

# ============================================================
# PAGE 5 — EXPLAINABLE AI (FIXED ONLY PERFORMANCE)
# ============================================================
elif page == "🧠 Explainable AI":

    st.title("🧠 Explainable AI Dashboard")

    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM machines LIMIT 200", conn)

    X = df.copy()

    for col in ["Target", "Predicted_Failure"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    st.subheader("📊 Global Feature Importance")

    importance_df = global_feature_importance(X)
    st.dataframe(importance_df, use_container_width=True)

    st.subheader("📈 SHAP Summary Plot")

    # ✅ ONLY CHANGE: ON DEMAND
    if st.button("⚡ Run SHAP Analysis"):
        with st.spinner("Computing SHAP values..."):
            shap_values, explainer = generate_shap_values(X)

            plt.figure()
            shap.summary_plot(shap_values, X, show=False)
            st.pyplot(plt.gcf())
            plt.clf()

# ============================================================
# PAGE 6 — Architecture (UNCHANGED)
# ============================================================
elif page == "🏗 Architecture":

    st.title("🏗 System Architecture")

    st.markdown("""
    ### System Flow

    1. User submits SQL query  
    2. Middleware analyzes complexity  
    3. Cost-based optimizer selects strategy  
    4. ML model injected (SQL / DL UDF / RL)  
    5. Query executed in SQLite  
    6. Results materialized  
    7. Feedback monitor evaluates health  
    """)

# Footer
st.divider()

if st.checkbox("🔍 Verify Materialized View"):
    conn = get_connection()
    try:
        m_df = pd.read_sql_query("SELECT * FROM predicted_results", conn)
        st.dataframe(m_df)
    except:
        st.info("No materialized table found yet.")