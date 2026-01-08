import threading
from pathlib import Path

import numpy as np
import xgboost as xgb

MODEL_PATH = Path("training/models/latest.json")

_model: xgb.Booster | None = None
_lock = threading.Lock()


def _load() -> xgb.Booster:
    if not MODEL_PATH.exists():
        raise RuntimeError("latest model not found")
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    return booster


def get_model() -> xgb.Booster:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = _load()
    return _model


def reload_model() -> None:
    global _model
    new_model = _load()
    with _lock:
        _model = new_model


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
        feature_names=FEATURE_COLUMNS,  # balls
    )

    prob = get_model().predict(dmatrix)[0]
    return float(prob)
