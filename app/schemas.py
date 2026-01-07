from pydantic import BaseModel


class Transaction(BaseModel):
    log_amount: float
    hour: int
    day: int
    is_night: int
    is_transfer: int
    is_cash_out: int
    type_freq: float
    tx_count_user: int
    avg_amount_user: float
    amount_vs_user_avg: float
    type_CASH_OUT: int
    type_DEBIT: int
    type_PAYMENT: int
    type_TRANSFER: int


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
