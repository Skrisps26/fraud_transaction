from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI

from app.model import get_model, predict_fraud, reload_model
from app.schemas import PredictionResponse, Transaction


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    get_model()
    yield
    # Shutdown (nothing to clean up)


app = FastAPI(lifespan=lifespan)


@app.post("/reload-model")
def reload():
    reload_model()
    return {"status": "reloaded"}


THRESHOLD = 0.9


@app.post("/predict")
def predict(transaction: Transaction):
    prob = predict_fraud(transaction.dict())

    decision = "BLOCK" if prob >= 0.85 else "REVIEW" if prob >= 0.7 else "ALLOW"

    return {"fraud_probability": prob, "decision": decision}
