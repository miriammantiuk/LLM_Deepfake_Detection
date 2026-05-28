# ---------------------------------------------------------
# metrics.py — Core evaluation metric functions
# ---------------------------------------------------------
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, roc_auc_score
)


def calculate_metrics(y_true, y_pred, y_prob=None):
    """Compute classification metrics for a single model/run.

    Returns a dict with Accuracy, Precision, Recall, F1, ROC AUC,
    and raw confusion matrix counts (TN, FP, FN, TP).
    All percentage values are multiplied by 100 and rounded to 1 decimal.
    """
    if len(y_true) == 0:
        return {
            'Accuracy (%)': 0, 'Precision (%)': 0, 'Recall (%)': 0,
            'F1-Score (%)': 0, 'ROC AUC': 'N/A',
            'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0
        }

    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall    = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1        = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    roc_auc = None
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 1:
                roc_auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            roc_auc = None

    return {
        'Accuracy (%)':  round(acc * 100, 1),
        'Precision (%)': round(precision * 100, 1),
        'Recall (%)':    round(recall * 100, 1),
        'F1-Score (%)':  round(f1 * 100, 1),
        'ROC AUC':       round(roc_auc, 1) if roc_auc is not None else 'N/A',
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }


def get_word_count(val):
    """Count words in a value that may be a string or a list of strings."""
    if isinstance(val, list):
        return len(" ".join(str(v) for v in val).split())
    elif isinstance(val, str):
        return len(val.split())
    return 0
