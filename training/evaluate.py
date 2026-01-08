# training/evaluate.py
from sklearn.metrics import roc_auc_score


def compute_auc(y_true, y_pred) -> float:
    return float(roc_auc_score(y_true, y_pred))
