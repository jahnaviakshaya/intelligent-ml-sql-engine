import sqlite3
import pandas as pd
import os

def setup():

    os.makedirs('db', exist_ok=True)
    db_path = os.path.join('db', 'intelligent_db.sqlite')

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # ==============================
    # 1️⃣ Load Predictive Maintenance Dataset
    # ==============================

    df = pd.read_csv("data/predictive_maintenance.csv")

    # Drop unused columns (same as training)
    cols_to_drop = [col for col in ["UDI", "Product ID", "Failure Type"] if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # One-hot encode (must match training exactly)
    df = pd.get_dummies(df, drop_first=True)

    # ==============================
    # 2️⃣ Store Data in SQLite
    # ==============================

    df.to_sql("machines", conn, if_exists="replace", index=False)

    # ==============================
    # 3️⃣ Users Table (RBAC)
    # ==============================

    cursor.execute("DROP TABLE IF EXISTS users")
    cursor.execute("""
        CREATE TABLE users (
            username TEXT PRIMARY KEY,
            role TEXT
        )
    """)

    users = [
        ('admin_user', 'Admin'),
        ('staff_user', 'Staff')
    ]

    cursor.executemany("INSERT INTO users VALUES (?, ?)", users)

    # ==============================
    # 4️⃣ Feedback Table
    # ==============================

    cursor.execute("DROP TABLE IF EXISTS model_feedback")
    cursor.execute("""
        CREATE TABLE model_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            correct INTEGER
        )
    """)

    # Mock feedback
    cursor.execute(
        "INSERT INTO model_feedback (correct) VALUES (1), (1), (0), (0), (1)"
    )

    conn.commit()
    conn.close()

    print(f"✅ Database initialized at {db_path}")
    print("✅ Machines table created with predictive maintenance data")

if __name__ == "__main__":
    setup()
