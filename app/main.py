"""
FastAPI Fraud Detection Service
Real-time fraud scoring endpoint
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pickle
from catboost import CatBoostClassifier
import numpy as np
from datetime import datetime
import sqlite3
from contextlib import asynccontextmanager

# Global model storage
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown"""
    print(" Loading fraud detection model...")
    
    # Determine which model to load
    try:
        with open('models/best_model.txt', 'r') as f:
            best_model = f.read().strip()
    except:
        best_model = 'CatBoost'  # default
    
    if best_model == 'CatBoost':
        model = CatBoostClassifier()
        model.load_model('models/catboost_fraud_model.cbm')
    else:
        with open('models/lightgbm_fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
    
    ml_models['fraud_model'] = model
    ml_models['model_name'] = best_model
    print(f" {best_model} model loaded successfully!")
    
    yield
    
    # Cleanup
    ml_models.clear()
    print(" Model unloaded")

# Create FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection using ML",
    version="1.0.0",
    lifespan=lifespan
)

# Request model
class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since first transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount in dollars", ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "Time": 0.0,
                "V1": -1.3598071336738,
                "V2": -0.0727811733098497,
                "V3": 2.53634673796914,
                "V4": 1.37815522427443,
                "V5": -0.338320769942518,
                "V6": 0.462387777762292,
                "V7": 0.239598554061257,
                "V8": 0.0986979012610507,
                "V9": 0.363786969611213,
                "V10": 0.0907941719789316,
                "V11": -0.551599533260813,
                "V12": -0.617800855762348,
                "V13": -0.991389847235408,
                "V14": -0.311169353699879,
                "V15": 1.46817697209427,
                "V16": -0.470400525259478,
                "V17": 0.207971241929242,
                "V18": 0.0257905801985591,
                "V19": 0.403992960255733,
                "V20": 0.251412098239705,
                "V21": -0.018306777944153,
                "V22": 0.277837575558899,
                "V23": -0.110473910188767,
                "V24": 0.0669280749146731,
                "V25": 0.128539358273528,
                "V26": -0.189114843888824,
                "V27": 0.133558376740387,
                "V28": -0.0210530534538215,
                "Amount": 149.62
            }
        }

# Response model
class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    fraud_flag: bool
    risk_level: str
    timestamp: str
    model_used: str

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Fraud Detection API",
        "model": ml_models.get('model_name', 'Unknown'),
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": 'fraud_model' in ml_models,
        "model_name": ml_models.get('model_name', 'Not loaded'),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: Transaction):
    """
    Predict fraud probability for a single transaction
    
    Returns fraud probability, binary flag, and risk level
    """
    try:
        # Get model
        if 'fraud_model' not in ml_models:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        model = ml_models['fraud_model']
        
        # Prepare features in correct order
        features = np.array([[
            transaction.Time,
            transaction.V1, transaction.V2, transaction.V3, transaction.V4, transaction.V5,
            transaction.V6, transaction.V7, transaction.V8, transaction.V9, transaction.V10,
            transaction.V11, transaction.V12, transaction.V13, transaction.V14, transaction.V15,
            transaction.V16, transaction.V17, transaction.V18, transaction.V19, transaction.V20,
            transaction.V21, transaction.V22, transaction.V23, transaction.V24, transaction.V25,
            transaction.V26, transaction.V27, transaction.V28,
            transaction.Amount
        ]])
        
        # Predict
        fraud_probability = float(model.predict_proba(features)[0][1])
        
        # Decision threshold (you can adjust this)
        threshold = 0.5
        fraud_flag = fraud_probability >= threshold
        
        # Risk level
        if fraud_probability < 0.3:
            risk_level = "LOW"
        elif fraud_probability < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Generate transaction ID
        transaction_id = f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        timestamp = datetime.now().isoformat()
        
        # Store in database
        conn = sqlite3.connect('data/fraud_predictions.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (
                timestamp, transaction_id, time_feature, amount,
                v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                v21, v22, v23, v24, v25, v26, v27, v28,
                fraud_probability, fraud_flag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp, transaction_id, transaction.Time, transaction.Amount,
            transaction.V1, transaction.V2, transaction.V3, transaction.V4, transaction.V5,
            transaction.V6, transaction.V7, transaction.V8, transaction.V9, transaction.V10,
            transaction.V11, transaction.V12, transaction.V13, transaction.V14, transaction.V15,
            transaction.V16, transaction.V17, transaction.V18, transaction.V19, transaction.V20,
            transaction.V21, transaction.V22, transaction.V23, transaction.V24, transaction.V25,
            transaction.V26, transaction.V27, transaction.V28,
            fraud_probability, int(fraud_flag)
        ))
        
        conn.commit()
        conn.close()
        
        # Return prediction
        return PredictionResponse(
            transaction_id=transaction_id,
            fraud_probability=round(fraud_probability, 4),
            fraud_flag=fraud_flag,
            risk_level=risk_level,
            timestamp=timestamp,
            model_used=ml_models['model_name']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/stats")
def get_statistics():
    """Get prediction statistics from database"""
    try:
        conn = sqlite3.connect('data/fraud_predictions.db')
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]
        
        # Fraud predictions
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE fraud_flag = 1")
        fraud_count = cursor.fetchone()[0]
        
        # Average fraud probability
        cursor.execute("SELECT AVG(fraud_probability) FROM predictions")
        avg_prob = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "total_predictions": total,
            "fraud_flagged": fraud_count,
            "fraud_rate": round(fraud_count / total * 100, 2) if total > 0 else 0,
            "average_fraud_probability": round(avg_prob, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)