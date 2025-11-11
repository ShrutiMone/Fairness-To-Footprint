from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
    false_positive_rate_difference,
    false_negative_rate_difference,
)
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


def compute_fairness_metrics(df, target_col, sensitive_col, pred_col=None):
    """
    Compute fairness metrics (overall + per sensitive group) using Fairlearn.
    Returns a structure like:
    {
        "overall": {metric_name: value, ...},
        "by_group": {
            "Male": {metric_name: value, ...},
            "Female": {metric_name: value, ...},
            ...
        }
    }
    """

    # --- Extract columns ---
    y_true = df[target_col]
    sensitive_feature = df[sensitive_col]

    # Use predicted column if available, otherwise assume perfect prediction
    y_pred = df[pred_col] if pred_col and pred_col in df.columns else y_true

    # --- Encode labels ---
    le = LabelEncoder()
    y_true_enc = le.fit_transform(y_true)
    y_pred_enc = le.transform(y_pred) if not y_pred.equals(y_true) else y_true_enc

    # Ensure binary labels
    unique_vals = np.unique(y_true_enc)
    if set(unique_vals) not in [{0, 1}, {-1, 1}]:
        y_true_enc = (y_true_enc == unique_vals.max()).astype(int)
        y_pred_enc = (y_pred_enc == unique_vals.max()).astype(int)

    # --- Define metrics ---
    metric_funcs = {
        "Selection Rate": selection_rate,
        "Demographic Parity Difference": demographic_parity_difference,
        "Equalized Odds Difference": equalized_odds_difference,
        "False Positive Rate Difference": false_positive_rate_difference,
        "False Negative Rate Difference": false_negative_rate_difference,
    }

    results = {"overall": {}, "by_group": {}}

    # --- Group-wise metric frame ---
    try:
        mf = MetricFrame(
            metrics={
                "Selection Rate": selection_rate,
                "False Positive Rate": lambda y_true, y_pred: (
                    ((y_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)
                ),
                "False Negative Rate": lambda y_true, y_pred: (
                    ((y_pred == 0) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)
                ),
            },
            y_true=y_true_enc,
            y_pred=y_pred_enc,
            sensitive_features=sensitive_feature,
        )

        # Store group-level metrics (selection rate, FPR, FNR)
        for group in mf.by_group.index:
            results["by_group"][str(group)] = {
                "Selection Rate": round(mf.by_group["Selection Rate"][group], 4),
                "False Positive Rate": round(mf.by_group["False Positive Rate"][group], 4),
                "False Negative Rate": round(mf.by_group["False Negative Rate"][group], 4),
            }

        # --- Compute global fairness differences ---
        results["overall"] = {
            "Demographic Parity Difference": round(
                demographic_parity_difference(
                    y_true=y_true_enc,
                    y_pred=y_pred_enc,
                    sensitive_features=sensitive_feature,
                ),
                4,
            ),
            "Equalized Odds Difference": round(
                equalized_odds_difference(
                    y_true=y_true_enc,
                    y_pred=y_pred_enc,
                    sensitive_features=sensitive_feature,
                ),
                4,
            ),
            "False Positive Rate Difference": round(
                false_positive_rate_difference(
                    y_true=y_true_enc,
                    y_pred=y_pred_enc,
                    sensitive_features=sensitive_feature,
                ),
                4,
            ),
            "False Negative Rate Difference": round(
                false_negative_rate_difference(
                    y_true=y_true_enc,
                    y_pred=y_pred_enc,
                    sensitive_features=sensitive_feature,
                ),
                4,
            ),
        }

    except Exception as e:
        results = {"error": str(e)}

    return results
