import numpy as np
import xgboost as xgb

booster = xgb.Booster()
booster.load_model("models/model_v1.json")

FEATURE_COLUMNS = [
    "log_amount",
    "hour",
    "day",
    "is_night",
    "is_transfer",
    "is_cash_out",
    "type_freq",
    "tx_count_user",
    "avg_amount_user",
    "amount_vs_user_avg",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]


def predict_fraud(features: dict) -> float:
    x = np.array([[features[c] for c in FEATURE_COLUMNS]], dtype=float)

    dmatrix = xgb.DMatrix(
        x,
        feature_names=FEATURE_COLUMNS,  # 👈 THIS IS THE KEY
    )

    prob = booster.predict(dmatrix)[0]
    return float(prob)
