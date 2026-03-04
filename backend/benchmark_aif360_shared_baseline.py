"""Run AIF360 mitigation benchmarks using the repo's shared baseline pipeline.

This script is designed to make Fairlearn-vs-AIF360 comparisons more credible by
reusing the SAME baseline training setup used in this repository:
- same feature prep strategy selection (build_transformer)
- same base classifier family by strategy
- same split protocol (80/20, random_state=42, stratified when possible)

Outputs JSON files under backend/benchmark_results by default.

Usage:
  cd backend
  python benchmark_aif360_shared_baseline.py --sensitive sex --constraint demographic_parity
  python benchmark_aif360_shared_baseline.py --sensitive sex --constraint equalized_odds
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_adult
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils.fairness_metrics import compute_fairness_metrics, compute_performance_metrics
from utils.mitigation import build_transformer


def _to_binary_target(y: pd.Series) -> pd.Series:
    ys = y.astype(str).str.strip().str.lower()
    positive = ys.str.contains('>50k') | ys.eq('1') | ys.eq('true')
    return positive.astype(int)


def _binary_sensitive(series: pd.Series):
    vals = pd.Series(series).astype(str).fillna("MISSING")
    uniq = sorted(vals.unique().tolist())
    if len(uniq) != 2:
        raise ValueError(
            f"AIF360 postprocessing in this script expects exactly 2 sensitive groups; got {len(uniq)}: {uniq}"
        )
    mapping = {uniq[0]: 0, uniq[1]: 1}
    return vals.map(mapping).astype(int).values, mapping


def _dataset_from_arrays(y_true, s_bin, label_name="label", sensitive_name="sensitive"):
    try:
        from aif360.datasets import BinaryLabelDataset
    except Exception as exc:
        raise RuntimeError(
            "AIF360 is required for this script. Install with: pip install aif360"
        ) from exc

    df = pd.DataFrame({label_name: y_true.astype(int), sensitive_name: s_bin.astype(int)})
    return BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df,
        label_names=[label_name],
        protected_attribute_names=[sensitive_name],
    )


def _apply_aif360_postprocessing(y_train, y_pred_train, y_score_train, s_train_bin,
                                 y_test, y_pred_test, y_score_test, s_test_bin,
                                 constraint: str):
    try:
        from aif360.algorithms.postprocessing import RejectOptionClassification
    except Exception as exc:
        raise RuntimeError(
            "AIF360 postprocessing not available. Install with: pip install aif360"
        ) from exc

    label_name = "label"
    sensitive_name = "sensitive"

    tr_true = _dataset_from_arrays(y_train, s_train_bin, label_name, sensitive_name)
    te_true = _dataset_from_arrays(y_test, s_test_bin, label_name, sensitive_name)

    tr_pred = tr_true.copy(deepcopy=True)
    tr_pred.labels = np.asarray(y_pred_train).reshape(-1, 1)
    tr_pred.scores = np.asarray(y_score_train).reshape(-1, 1)

    te_pred = te_true.copy(deepcopy=True)
    te_pred.labels = np.asarray(y_pred_test).reshape(-1, 1)
    te_pred.scores = np.asarray(y_score_test).reshape(-1, 1)

    # AIF360 uses these metric names for ROC search.
    metric_name = "Statistical parity difference" if constraint == "demographic_parity" else "Average odds difference"

    rop = RejectOptionClassification(
        unprivileged_groups=[{sensitive_name: 0}],
        privileged_groups=[{sensitive_name: 1}],
        low_class_thresh=0.01,
        high_class_thresh=0.99,
        num_class_thresh=100,
        num_ROC_margin=50,
        metric_name=metric_name,
        metric_ub=0.05,
        metric_lb=-0.05,
    )
    rop.fit(tr_true, tr_pred)
    te_mitigated = rop.predict(te_pred)
    return te_mitigated.labels.ravel().astype(int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitive", default="sex", help="Sensitive attribute column: sex or race")
    parser.add_argument("--constraint", default="demographic_parity", choices=["demographic_parity", "equalized_odds"])
    parser.add_argument("--output_dir", default="benchmark_results")
    parser.add_argument("--csv_path", default="", help="Optional local Adult CSV path for offline runs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    # Load data
    if args.csv_path:
        log(f"[aif360-benchmark] Loading local CSV: {args.csv_path}")
        df = pd.read_csv(args.csv_path)
        if "income" not in df.columns:
            raise ValueError("Local CSV must include an 'income' target column.")
        df["income"] = _to_binary_target(df["income"])
    else:
        log("[aif360-benchmark] Downloading Adult dataset via fairlearn.datasets.fetch_adult ...")
        ds = fetch_adult(as_frame=True)
        X = ds.data.copy()
        y = _to_binary_target(pd.Series(ds.target, name="income"))
        if args.sensitive not in X.columns:
            raise ValueError(f"Sensitive column '{args.sensitive}' not in dataset columns")
        df = X.copy()
        df["income"] = y

    # Shared baseline setup (same logic used in repo baseline)
    transformer, strategy, _te = build_transformer(df, target_col="income", sensitive_col=args.sensitive)
    if strategy == "fast-hash-sgd":
        base_clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
    else:
        base_clf = LogisticRegression(max_iter=2000, solver='saga', n_jobs=-1, random_state=42)

    X_all = df.drop(columns=["income", args.sensitive], errors="ignore")
    y_all = df["income"].astype(int).values
    s_all_raw = df[args.sensitive]

    try:
        X_train, X_test, y_train, y_test, s_train_raw, s_test_raw = train_test_split(
            X_all, y_all, s_all_raw, test_size=0.2, random_state=42, stratify=y_all
        )
    except Exception:
        X_train, X_test, y_train, y_test, s_train_raw, s_test_raw = train_test_split(
            X_all, y_all, s_all_raw, test_size=0.2, random_state=42
        )

    baseline_pipeline = Pipeline(steps=[("pre", transformer), ("clf", base_clf)])
    baseline_pipeline.fit(X_train, y_train)

    y_pred_train = baseline_pipeline.predict(X_train).astype(int)
    y_pred_test = baseline_pipeline.predict(X_test).astype(int)

    # Scores for AIF360 postprocessing
    if hasattr(baseline_pipeline, "predict_proba"):
        y_score_train = baseline_pipeline.predict_proba(X_train)[:, 1]
        y_score_test = baseline_pipeline.predict_proba(X_test)[:, 1]
    else:
        y_score_train = y_pred_train.astype(float)
        y_score_test = y_pred_test.astype(float)

    s_train_bin, mapping = _binary_sensitive(pd.Series(s_train_raw))
    s_test_bin = pd.Series(s_test_raw).astype(str).fillna("MISSING").map(mapping).astype(int).values

    y_pred_test_mitigated = _apply_aif360_postprocessing(
        y_train, y_pred_train, y_score_train, s_train_bin,
        y_test, y_pred_test, y_score_test, s_test_bin,
        constraint=args.constraint,
    )

    # Compute metrics using same local metric functions/labels as the app
    tmp_base = pd.DataFrame(X_test).copy()
    tmp_base["income"] = y_test
    tmp_base[args.sensitive] = pd.Series(s_test_raw).values
    tmp_base["y_pred"] = y_pred_test

    tmp_mit = pd.DataFrame(X_test).copy()
    tmp_mit["income"] = y_test
    tmp_mit[args.sensitive] = pd.Series(s_test_raw).values
    tmp_mit["y_pred"] = y_pred_test_mitigated

    baseline_fair = compute_fairness_metrics(tmp_base, "income", args.sensitive, pred_col="y_pred")
    baseline_perf = compute_performance_metrics(y_test, y_pred_test)

    mitigated_fair = compute_fairness_metrics(tmp_mit, "income", args.sensitive, pred_col="y_pred")
    mitigated_perf = compute_performance_metrics(y_test, y_pred_test_mitigated)

    result = {
        "dataset": "fairlearn.datasets.fetch_adult" if not args.csv_path else args.csv_path,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target": "income",
        "sensitive": args.sensitive,
        "constraint": args.constraint,
        "library": "AIF360",
        "baseline_strategy": strategy,
        "sensitive_mapping": mapping,
        "baseline": {
            "performance_test": baseline_perf,
            "fairness_test": baseline_fair.get("overall", {}),
            "fairness_test_by_group": baseline_fair.get("by_group", {}),
        },
        "aif360_mitigated": {
            "performance_test": mitigated_perf,
            "fairness_test": mitigated_fair.get("overall", {}),
            "fairness_test_by_group": mitigated_fair.get("by_group", {}),
        },
    }

    out_json = out_dir / f"adult_{args.sensitive}_aif360_{args.constraint}.json"
    out_json.write_text(json.dumps(result, indent=2))
    log(f"[aif360-benchmark] Saved: {out_json}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[aif360-benchmark] ERROR: {exc}", file=sys.stderr, flush=True)
        raise
