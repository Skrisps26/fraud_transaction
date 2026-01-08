# training/retrain.py
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb
import yaml
from evaluate import compute_auc
from features import FEATURE_COLUMNS, build_features

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_data(path):
    df = pd.read_csv(path)
    df = df[df["type"].isin(["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT"])]
    return df


def time_split(df, test_days):
    max_day = df["step"].max() // 24
    cutoff = (max_day - test_days) * 24
    train = df[df["step"] < cutoff]
    val = df[df["step"] >= cutoff]
    return train, val


def load_previous_auc():
    auc_file = MODELS_DIR / "latest_auc.txt"
    if not auc_file.exists():
        return 0.0
    return float(auc_file.read_text().strip())


def next_model_version():
    existing = MODELS_DIR.glob("model_v*.json")
    versions = [
        int(p.stem.replace("model_v", ""))
        for p in existing
        if p.stem.replace("model_v", "").isdigit()
    ]
    return max(versions, default=0) + 1


def main():
    config = load_config()

    df = load_data(config["data"]["path"])
    train_df, val_df = time_split(df, config["training"]["test_days"])

    X_train = build_features(pd.DataFrame(train_df))
    y_train = train_df["isFraud"]

    X_val = build_features(pd.DataFrame(val_df))
    y_val = val_df["isFraud"]

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLUMNS)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLUMNS)

    params = config["xgboost"]
    num_rounds = params.pop("num_boost_round")

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
    )

    preds = booster.predict(dval)
    new_auc = compute_auc(y_val, preds)

    old_auc = load_previous_auc()
    improvement = new_auc - old_auc

    print(f"Old AUC: {old_auc:.4f}")
    print(f"New AUC: {new_auc:.4f}")
    print(f"Δ AUC: {improvement:.4f}")

    if improvement < config["training"]["min_auc_improvement"]:
        print("❌ Model not good enough. Skipping promotion.")
        sys.exit(1)

    # ----- Promote -----
    version = next_model_version()
    model_path = MODELS_DIR / f"model_v{version}.json"
    booster.save_model(model_path)

    latest_path = MODELS_DIR / "latest.json"
    booster.save_model(latest_path)

    (MODELS_DIR / "latest_auc.txt").write_text(f"{new_auc:.6f}")

    print(f"✅ Promoted model_v{version}")
    sys.exit(0)


if __name__ == "__main__":
    main()
