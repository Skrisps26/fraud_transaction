from fastapi import FastAPI

from app.model import predict_fraud
from app.schemas import PredictionResponse, Transaction

app = FastAPI(
    title="Fraud Risk Scoring API",
    description="XGBoost-based fraud detection model",
    version="1.0",
)

THRESHOLD = 0.9


@app.post("/predict")
def predict(transaction: Transaction):
    prob = predict_fraud(transaction.dict())

    decision = "BLOCK" if prob >= 0.85 else "REVIEW" if prob >= 0.7 else "ALLOW"

    return {"fraud_probability": prob, "decision": decision}
