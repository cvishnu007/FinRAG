from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    unique_labels = set(y_true)
    if len(unique_labels) < 2:
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, y_proba))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
    }


def print_metrics(label: str, metrics: Dict[str, float]) -> None:
    print(f"\n{label} metrics")
    print("-" * (len(label) + 8))
    for k, v in metrics.items():
        print(f"{k:>9}: {v:.4f}")
