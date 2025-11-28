# backend/utils/fairness_metrics.py
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate_difference,
    false_negative_rate_difference,
)
from fairlearn.metrics import selection_rate
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

def _ensure_binary(y):
    y = np.asarray(y)
    vals = np.unique(y)
    if set(vals) <= {0, 1} or set(vals) <= {-1, 1}:
        return y.astype(int)
    # map max to 1, others to 0
    return (y == vals.max()).astype(int)

def compute_fairness_metrics(df: pd.DataFrame, target_col: str, sensitive_col: str, pred_col: str = None):
    """
    Return {"overall": {...}, "by_group": {...}} or {"error": "..."}
    """
    try:
        if target_col not in df.columns:
            return {"error": f"target column '{target_col}' not found"}
        if sensitive_col not in df.columns:
            return {"error": f"sensitive column '{sensitive_col}' not found"}

        y_true = df[target_col]
        y_pred = df[pred_col] if pred_col and pred_col in df.columns else y_true
        sensitive = df[sensitive_col]

        # encode labels if object dtype
        if y_true.dtype == object or y_pred.dtype == object:
            le = LabelEncoder()
            y_true_enc = le.fit_transform(y_true)
            try:
                y_pred_enc = le.transform(y_pred)
            except Exception:
                y_pred_enc = LabelEncoder().fit_transform(y_pred)
        else:
            y_true_enc = y_true.values.astype(int)
            y_pred_enc = y_pred.values.astype(int)

        y_true_enc = _ensure_binary(y_true_enc)
        y_pred_enc = _ensure_binary(y_pred_enc)

        sensitive_series = pd.Series(sensitive).fillna("MISSING")

        # MetricFrame for selection rate, FPR, FNR groupwise
        mf = MetricFrame(
            metrics={
                "Selection Rate": selection_rate,
                "False Positive Rate": lambda y_t, y_p: ((y_p == 1) & (y_t == 0)).sum() / max(((y_t == 0).sum()), 1),
                "False Negative Rate": lambda y_t, y_p: ((y_p == 0) & (y_t == 1)).sum() / max(((y_t == 1).sum()), 1),
            },
            y_true=y_true_enc,
            y_pred=y_pred_enc,
            sensitive_features=sensitive_series,
        )

        overall = {
            "Demographic Parity Difference": round(float(demographic_parity_difference(
                y_true=y_true_enc, y_pred=y_pred_enc, sensitive_features=sensitive_series)), 4),
            "Equalized Odds Difference": round(float(equalized_odds_difference(
                y_true=y_true_enc, y_pred=y_pred_enc, sensitive_features=sensitive_series)), 4),
            "False Positive Rate Difference": round(float(false_positive_rate_difference(
                y_true=y_true_enc, y_pred=y_pred_enc, sensitive_features=sensitive_series)), 4),
            "False Negative Rate Difference": round(float(false_negative_rate_difference(
                y_true=y_true_enc, y_pred=y_pred_enc, sensitive_features=sensitive_series)), 4),
        }

        by_group = {}
        for g in mf.by_group.index:
            sel = mf.by_group["Selection Rate"].get(g, np.nan)
            fpr = mf.by_group["False Positive Rate"].get(g, np.nan)
            fnr = mf.by_group["False Negative Rate"].get(g, np.nan)
            by_group[str(g)] = {
                "Selection Rate": round(float(sel), 4) if not np.isnan(sel) else None,
                "False Positive Rate": round(float(fpr), 4) if not np.isnan(fpr) else None,
                "False Negative Rate": round(float(fnr), 4) if not np.isnan(fnr) else None,
            }

        return {"overall": overall, "by_group": by_group}
    except Exception as e:
        return {"error": str(e)}
