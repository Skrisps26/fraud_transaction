from fastapi import FastAPI
from app.schemas import Transaction, PredictionResponse
from app.model import predict_fraud

app = FastAPI(
    title="Fraud Risk Scoring API",
    description="XGBoost-based fraud detection model",
    version="1.0"
)

THRESHOLD = 0.9

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    prob = predict_fraud(transaction.dict())
    return {
        "fraud_probability": prob,
        "is_fraud": prob >= THRESHOLD
    }
