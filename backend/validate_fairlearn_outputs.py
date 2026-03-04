"""Validate Fairlearn baseline/mitigation outputs with manual metric calculations.

This script is meant for Results & Discussion reproducibility.
It recomputes fairness metrics manually from predictions and compares them
against values returned by the project's Fairlearn-based pipeline.

Usage:
  cd backend
  python validate_fairlearn_outputs.py --sensitive sex --constraint demographic_parity
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_adult

from utils.mitigation import mitigate_with_exponentiated_gradient


def _to_binary_target(y: pd.Series) -> pd.Series:
    ys = y.astype(str).str.strip().str.lower()
    positive = ys.str.contains(">50k") | ys.eq("1") | ys.eq("true")
    return positive.astype(int)


def _manual_group_rates(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray):
    groups = pd.Series(sensitive).astype(str).fillna("MISSING").unique().tolist()
    by_group = {}

    for g in groups:
        mask = pd.Series(sensitive).astype(str).fillna("MISSING").values == g
        yg = y_true[mask]
        pg = y_pred[mask]

        selection_rate = float((pg == 1).mean()) if len(pg) else np.nan
        negatives = max(int((yg == 0).sum()), 1)
        positives = max(int((yg == 1).sum()), 1)
        fpr = float(((pg == 1) & (yg == 0)).sum() / negatives)
        fnr = float(((pg == 0) & (yg == 1)).sum() / positives)

        by_group[g] = {
            "Selection Rate": selection_rate,
            "False Positive Rate": fpr,
            "False Negative Rate": fnr,
        }

    return by_group


def _manual_overall_differences(by_group: dict):
    sel = [v["Selection Rate"] for v in by_group.values()]
    fpr = [v["False Positive Rate"] for v in by_group.values()]
    fnr = [v["False Negative Rate"] for v in by_group.values()]

    dp_diff = float(np.nanmax(sel) - np.nanmin(sel))
    fpr_diff = float(np.nanmax(fpr) - np.nanmin(fpr))
    fnr_diff = float(np.nanmax(fnr) - np.nanmin(fnr))
    eo_diff = max(fpr_diff, fnr_diff)

    return {
        "Demographic Parity Difference": round(dp_diff, 4),
        "Equalized Odds Difference": round(eo_diff, 4),
        "False Positive Rate Difference": round(fpr_diff, 4),
        "False Negative Rate Difference": round(fnr_diff, 4),
    }


def _assert_close(a: dict, b: dict, tol: float = 1e-4):
    for k in a:
        if abs(float(a[k]) - float(b[k])) > tol:
            raise AssertionError(f"Mismatch for {k}: manual={a[k]} fairlearn={b[k]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitive", default="sex", help="Sensitive attribute column")
    parser.add_argument("--constraint", default="demographic_parity", choices=["demographic_parity", "equalized_odds"])
    parser.add_argument("--out", default="benchmark_results/validation_check.json")
    parser.add_argument("--csv_path", default="", help="Optional local Adult CSV for offline runs")
    args = parser.parse_args()

    if args.csv_path:
        df = pd.read_csv(args.csv_path)
        if "income" not in df.columns:
            raise ValueError("CSV must include an 'income' target column.")
        df["income"] = _to_binary_target(df["income"])
    else:
        ds = fetch_adult(as_frame=True)
        X = ds.data.copy()
        y = _to_binary_target(pd.Series(ds.target, name="income"))

        if args.sensitive not in X.columns:
            raise ValueError(f"Sensitive column '{args.sensitive}' not found in Adult columns")

        df = X.copy()
        df["income"] = y

    res = mitigate_with_exponentiated_gradient(
        df,
        target_col="income",
        sensitive_col=args.sensitive,
        constraint=args.constraint,
    )

    # Recreate same holdout split used by helper using known random_state/stratify.
    from sklearn.model_selection import train_test_split

    X_all = df.drop(columns=["income", args.sensitive], errors="ignore")
    y_all = df["income"].astype(int).values
    s_all = df[args.sensitive].values
    _, X_test, _, y_test, _, s_test = train_test_split(
        X_all, y_all, s_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # Recover predictors used in helper.
    # Baseline pipeline is first predictor in mitigator internals not guaranteed,
    # so we directly use helper-reported fairness for baseline and validate formulas
    # with mitigated predictions here.
    y_pred_mitigated_test = res.get("predictions_mitigated_test")
    if y_pred_mitigated_test is None:
        mitigator = res["mitigator"]
        y_pred_mitigated_test = mitigator.predict(X_test)
    y_pred_mitigated_test = np.asarray(y_pred_mitigated_test)

    manual_by_group = _manual_group_rates(y_test, y_pred_mitigated_test, s_test)
    manual_overall = _manual_overall_differences(manual_by_group)

    fair_overall = res["metrics_after_mitigation_test"]["overall"]
    _assert_close(manual_overall, fair_overall)

    out = {
        "sensitive": args.sensitive,
        "constraint": args.constraint,
        "manual_overall": manual_overall,
        "fairlearn_overall": fair_overall,
        "status": "pass",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
