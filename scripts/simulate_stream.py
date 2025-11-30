"""
Streaming Transaction Simulator - FIXED VERSION
Simulates real-time fraud detection by sending transactions to API
NOW WITH PROPER TRUE LABEL TRACKING FOR ACCURATE METRICS
"""
import pandas as pd
import requests
import time
import json
from tqdm import tqdm
from datetime import datetime
import sqlite3

# Configuration
API_URL = "http://127.0.0.1:8000/predict"
SLEEP_TIME = 0.05  # seconds between transactions (50ms)
NUM_TRANSACTIONS = 1000  # How many transactions to stream

print("="*80)
print("STREAMING TRANSACTION SIMULATOR - FIXED VERSION")
print("="*80)

# Load test dataset
print("\nLoading transaction data...")
df = pd.read_csv('data/creditcard.csv')

# Use the last 20% as "streaming" data (since we trained on first 80%)
split_index = int(len(df) * 0.8)
streaming_data = df.iloc[split_index:split_index + NUM_TRANSACTIONS].copy()

print(f"Loaded {len(streaming_data)} transactions for streaming")
print(f"   Fraud cases in stream: {streaming_data['Class'].sum()}")
print(f"   Normal cases in stream: {(streaming_data['Class'] == 0).sum()}")

# Check if API is running
print("\nChecking API connection...")
try:
    response = requests.get("http://127.0.0.1:8000/health")
    if response.status_code == 200:
        print("API is running and healthy!")
    else:
        print("ERROR: API responded with error")
        exit(1)
except:
    print("ERROR: Cannot connect to API!")
    print("   Please start the API first with: uvicorn app.main:app --reload")
    exit(1)

# Stream transactions
print(f"\nStarting to stream {NUM_TRANSACTIONS} transactions...")
print(f"   Rate: 1 transaction every {SLEEP_TIME} seconds")
print(f"   Estimated time: {NUM_TRANSACTIONS * SLEEP_TIME / 60:.1f} minutes")
print()

results = []
start_time = datetime.now()

for idx, row in tqdm(streaming_data.iterrows(), total=len(streaming_data), desc="Streaming"):
    # Prepare transaction data
    transaction = {
        "Time": float(row['Time']),
        "V1": float(row['V1']),
        "V2": float(row['V2']),
        "V3": float(row['V3']),
        "V4": float(row['V4']),
        "V5": float(row['V5']),
        "V6": float(row['V6']),
        "V7": float(row['V7']),
        "V8": float(row['V8']),
        "V9": float(row['V9']),
        "V10": float(row['V10']),
        "V11": float(row['V11']),
        "V12": float(row['V12']),
        "V13": float(row['V13']),
        "V14": float(row['V14']),
        "V15": float(row['V15']),
        "V16": float(row['V16']),
        "V17": float(row['V17']),
        "V18": float(row['V18']),
        "V19": float(row['V19']),
        "V20": float(row['V20']),
        "V21": float(row['V21']),
        "V22": float(row['V22']),
        "V23": float(row['V23']),
        "V24": float(row['V24']),
        "V25": float(row['V25']),
        "V26": float(row['V26']),
        "V27": float(row['V27']),
        "V28": float(row['V28']),
        "Amount": float(row['Amount'])
    }
    
    try:
        # Send to API
        response = requests.post(API_URL, json=transaction)
        
        if response.status_code == 200:
            pred = response.json()
            
            # Store with TRUE LABEL
            results.append({
                'transaction_id': pred['transaction_id'],
                'true_label': int(row['Class']),  # THIS IS THE FIX!
                'fraud_probability': pred['fraud_probability'],
                'fraud_flag': int(pred['fraud_flag']),
                'risk_level': pred['risk_level']
            })
            
            # ALSO UPDATE DATABASE WITH TRUE LABEL
            conn = sqlite3.connect('data/fraud_predictions.db')
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE predictions 
                SET true_label = ? 
                WHERE transaction_id = ?
            ''', (int(row['Class']), pred['transaction_id']))
            conn.commit()
            conn.close()
            
        else:
            print(f"\nERROR on transaction {idx}: {response.status_code}")
    
    except Exception as e:
        print(f"\nERROR: {str(e)}")
    
    # Simulate streaming delay
    time.sleep(SLEEP_TIME)

end_time = datetime.now()
duration = (end_time - start_time).total_seconds()

# Analyze results
print("\n" + "="*80)
print("STREAMING RESULTS")
print("="*80)

results_df = pd.DataFrame(results)

# True Positives, False Positives, etc.
tp = ((results_df['true_label'] == 1) & (results_df['fraud_flag'] == 1)).sum()
fp = ((results_df['true_label'] == 0) & (results_df['fraud_flag'] == 1)).sum()
tn = ((results_df['true_label'] == 0) & (results_df['fraud_flag'] == 0)).sum()
fn = ((results_df['true_label'] == 1) & (results_df['fraud_flag'] == 0)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nStreaming Duration: {duration:.2f} seconds")
print(f"Throughput: {len(results) / duration:.2f} transactions/second")
print(f"\nPredictions Made: {len(results)}")
print(f"   Flagged as Fraud: {results_df['fraud_flag'].sum()}")
print(f"   Flagged as Normal: {(results_df['fraud_flag'] == 0).sum()}")
print(f"\nActual Labels:")
print(f"   Actual Fraud: {results_df['true_label'].sum()}")
print(f"   Actual Normal: {(results_df['true_label'] == 0).sum()}")
print(f"\nPerformance Metrics:")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"\nConfusion Matrix:")
print(f"   True Positives:  {tp} (Correctly identified fraud)")
print(f"   False Positives: {fp} (Normal flagged as fraud)")
print(f"   True Negatives:  {tn} (Correctly identified normal)")
print(f"   False Negatives: {fn} (Fraud missed)")
print(f"\nRisk Level Distribution:")
print(results_df['risk_level'].value_counts())

# Business metrics
if tp + fn > 0:
    fraud_caught_rate = (tp / (tp + fn)) * 100
    print(f"\nBusiness Impact:")
    print(f"   Fraud Detection Rate: {fraud_caught_rate:.2f}%")
    print(f"   Frauds Caught: {tp} out of {tp + fn}")
    print(f"   Frauds Missed: {fn}")

if tp + fp > 0:
    investigation_accuracy = (tp / (tp + fp)) * 100
    print(f"   Investigation Accuracy: {investigation_accuracy:.2f}%")
    print(f"   (Of all flagged cases, {investigation_accuracy:.2f}% are actually fraud)")

# Save results
results_df.to_csv('data/streaming_results.csv', index=False)
print(f"\nResults saved to: data/streaming_results.csv")

print("\n" + "="*80)
print("STREAMING SIMULATION COMPLETE")
print("="*80)
print("\nNext Step: Refresh your Streamlit dashboard to see updated metrics!")
print("   The dashboard will now show correct Precision, Recall, and F1-Score!")