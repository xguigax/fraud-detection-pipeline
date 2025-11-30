# Fraud Detection Pipeline

An end-to-end ML system for detecting credit card fraud. Includes model training, a FastAPI server for predictions, and a Streamlit dashboard for monitoring.

## What This Is

I built this to learn how ML systems work in production, not just in notebooks. The project takes a credit card fraud dataset, trains gradient boosting models on it, serves predictions through an API, and monitors everything with a dashboard.

The best model (CatBoost) gets 99.1% ROC-AUC and 79.8% PR-AUC on test data. More importantly, it handles the massive class imbalance (only 0.17% of transactions are fraud) without just predicting "not fraud" for everything.

## Why Fraud Detection

Fraud is a classic imbalanced classification problem - perfect for learning how to handle real-world ML challenges. Plus it has clear business metrics (money saved, frauds caught) rather than just accuracy scores.

I wanted something I could actually show to companies like WTW or Aon and have a real conversation about the engineering decisions, not just "here's my Kaggle notebook."

## Tech Stack

**Models:** CatBoost, LightGBM  
**API:** FastAPI with Pydantic validation  
**Database:** SQLite for logging predictions  
**Dashboard:** Streamlit with Plotly charts  
**Other:** scikit-learn, pandas, imbalanced-learn

## How It Works

The pipeline is: train models on historical data, serve the best one through FastAPI, simulate real-time transactions hitting the API, log everything to SQLite, and visualize it all in a Streamlit dashboard.

## Results

### Model Comparison

| Model | ROC-AUC | PR-AUC | F1-Score |
|-------|---------|--------|----------|
| CatBoost | 0.9910 | 0.7980 | 0.3769 |
| LightGBM | 0.8610 | 0.0070 | 0.0155 |

CatBoost wins by a lot. LightGBM completely fails on the minority class (fraud), which you can see from the terrible PR-AUC.

### Why PR-AUC Matters More Than ROC-AUC

With 99.83% of transactions being normal, you can get 99.8% accuracy by just always predicting "not fraud." ROC-AUC can be misleading because it's inflated by the huge number of true negatives.

PR-AUC only looks at the positive class (fraud), so it's a much better metric here. The 79.8% PR-AUC means the model is actually good at finding fraud without flagging everything.

## What's In Here

### Data Exploration
- Analyzed all 284K transactions
- Found fraud is 0.17% of data (577:1 imbalance)
- Checked distributions, correlations, missing values
- Made some visualizations to understand the data

### Model Training
- Split data by time (train on early transactions, test on later ones) - more realistic than random split
- Used `scale_pos_weight` in CatBoost to handle imbalance
- Compared models on multiple metrics
- Saved the best one

### FastAPI Server
- `/predict` endpoint takes transaction features, returns fraud probability
- `/health` for checking if it's running
- `/stats` shows how many predictions made, fraud rate, etc.
- Logs everything to SQLite
- Auto-generated docs at `/docs`

### Streaming Simulator
- Reads test data and sends it to the API one transaction at a time
- Simulates what real-time scoring would look like
- Measures throughput (about 20 predictions/second)

### Dashboard (Streamlit)
Has 4 tabs:

**Overview** - Total transactions, fraud detected, probability distribution, high-risk alerts

**Model Performance** - Shows CatBoost vs LightGBM comparison with charts

**Real-Time Monitoring** - Fraud rate over time, average amounts, probability trends

**Analysis** - Amount vs probability scatter plot, fraud risk by transaction size

## Installation

You need Python 3.12 or newer.

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-pipeline.git
cd fraud-detection-pipeline

python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

Get the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and put `creditcard.csv` in the `data/` folder.

## Running It

**Train models:**
```bash
python scripts/train_models.py
```
Takes 5-7 minutes. Creates model files in `models/`.

**Set up database:**
```bash
python scripts/init_database.py
```

**Start the API:**
```bash
uvicorn app.main:app --reload
```
Go to http://127.0.0.1:8000/docs to see the API documentation.

**Run the simulator** (in a new terminal):
```bash
python scripts/simulate_stream.py
```
Sends 1000 transactions to the API. Takes about a minute.

**Launch dashboard** (in another new terminal):
```bash
streamlit run dashboard.py
```
Opens automatically at http://localhost:8501

## File Structure

**data/** - Dataset and databases  
**models/** - Trained model files and metrics  
**app/** - FastAPI server code  
**scripts/** - Training, database setup, simulator  
**notebooks/** - Jupyter notebook for EDA  
**figures/** - Charts and visualizations  
**dashboard.py** - Streamlit monitoring app

## Using the API

**Make a prediction:**
```bash
POST /predict
```
Send JSON with Time, V1-V28, and Amount. Get back fraud probability and risk level.

**Check health:**
```bash
GET /health
```

**Get stats:**
```bash
GET /stats
```

## Technical Decisions

### Why CatBoost Over LightGBM

LightGBM is faster, but CatBoost handles imbalanced data way better. You can see this in the PR-AUC difference (79.8% vs 0.7%). For fraud detection, that's the whole ballgame.

### Handling Imbalance

I tried a few approaches:
- SMOTE (synthetic oversampling) - made things worse
- Class weights with `scale_pos_weight` - worked great
- Adjusting prediction threshold - helps but not as much as proper weights

The key insight: don't try to balance the data, make the model care more about the minority class.

### Time-Based Split

Random train/test split is unrealistic for time-series data. In production you train on the past and predict the future. So I split chronologically - first 80% for training, last 20% for testing.

### Why SQLite

For a portfolio project, SQLite is perfect. It's just a file, no server needed. In production you'd use Postgres or something, but this works fine for logging predictions and learning how it all fits together.

## Business Impact

If you assume:
- 10,000 transactions/year
- 2% fraud rate (200 frauds)
- $15K average fraud amount
- Baseline catches 60% of fraud

Then this model at 85% recall would catch an extra 50 frauds per year = $750K saved.

Obviously these numbers are made up for the example dataset, but the point is you can translate model metrics into dollars. That's what actuarial firms care about.

## Things I Learned

**PR-AUC > ROC-AUC for imbalanced data.** This was a big one. ROC-AUC looks great but doesn't tell you if the model actually catches fraud.

**Default threshold (0.5) is usually wrong.** You need to pick a threshold based on business constraints - how much fraud can you tolerate missing vs how many false alarms can investigators handle.

**Production is different than notebooks.** Error handling, logging, monitoring, API design - none of that matters in a Jupyter notebook but it's most of the work in production.

**Class weights > oversampling for tree models.** SMOTE made my models worse. Just using scale_pos_weight was better and faster.

## What I'd Add Next

- SHAP values to explain individual predictions
- Automated drift detection with Evidently
- Docker container to make deployment easier
- Real A/B testing between models
- Threshold optimization based on actual costs (not just F1)
- Model retraining pipeline when performance drops

## Dataset

Credit Card Fraud Detection dataset from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

284,807 transactions from September 2013  
492 frauds (0.17%)  
Features are PCA-transformed for privacy  

## About

Built by Gabriel [Last Name]

This was a portfolio project to learn production ML and have something concrete to discuss in interviews for actuarial/risk analyst roles.

GitHub: github.com/your-username  
LinkedIn: linkedin.com/in/your-profile  
Email: your.email@example.com