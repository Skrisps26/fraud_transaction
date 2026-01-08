# training/features.py
import numpy as np
import pandas as pd

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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ----- Amount -----
    df["log_amount"] = np.log1p(df["amount"])

    # ----- Time -----
    df["hour"] = df["step"] % 24
    df["day"] = df["step"] // 24
    df["is_night"] = df["hour"].between(0, 5).astype(int)

    # ----- Transaction type flags -----
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    df["is_cash_out"] = (df["type"] == "CASH_OUT").astype(int)

    # ----- User behavior -----
    df["tx_count_user"] = df.groupby("nameOrig")["step"].transform("count")
    df["avg_amount_user"] = df.groupby("nameOrig")["amount"].transform("mean")
    df["amount_vs_user_avg"] = df["amount"] / (df["avg_amount_user"] + 1e-6)

    # ----- Type frequency -----
    type_freq = df["type"].value_counts(normalize=True).to_dict()
    df["type_freq"] = df["type"].replace(type_freq)

    # ----- One-hot (explicit order) -----
    for t in ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        df[f"type_{t}"] = (df["type"] == t).astype(int)

    return pd.DataFrame(df[FEATURE_COLUMNS])
