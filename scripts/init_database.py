"""
Initialize SQLite database for fraud predictions
"""
import sqlite3
from datetime import datetime

print("🗄️  Creating fraud predictions database...")

# Connect to database (creates file if doesn't exist)
conn = sqlite3.connect('data/fraud_predictions.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    time_feature REAL,
    amount REAL,
    v1 REAL, v2 REAL, v3 REAL, v4 REAL, v5 REAL,
    v6 REAL, v7 REAL, v8 REAL, v9 REAL, v10 REAL,
    v11 REAL, v12 REAL, v13 REAL, v14 REAL, v15 REAL,
    v16 REAL, v17 REAL, v18 REAL, v19 REAL, v20 REAL,
    v21 REAL, v22 REAL, v23 REAL, v24 REAL, v25 REAL,
    v26 REAL, v27 REAL, v28 REAL,
    fraud_probability REAL NOT NULL,
    fraud_flag INTEGER NOT NULL,
    true_label INTEGER
)
''')

conn.commit()
conn.close()

print(" Database created: data/fraud_predictions.db")
print(" Table 'predictions' ready to store transaction scores")