import json
import numpy as np
import pandas as pd

from fairlearn.datasets import fetch_adult
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing import EqOddsPostprocessing


def to_binary_income(y):
    ys = pd.Series(y).astype(str).str.strip().str.lower()
    return (ys.str.contains(">50k") | ys.eq("1") | ys.eq("true")).astype(int).values


def fairness_diffs(y_true, y_pred, sensitive):
    # same style as your Fairlearn reporting columns
    groups = pd.Series(sensitive).astype(str).fillna("MISSING").unique().tolist()

    by_group = {}
    for g in groups:
        m = (pd.Series(sensitive).astype(str).fillna("MISSING").values == g)
        yt = y_true[m]
        yp = y_pred[m]
        sel = (yp == 1).mean() if len(yp) else np.nan
        neg = max((yt == 0).sum(), 1)
        pos = max((yt == 1).sum(), 1)
        fpr = ((yp == 1) & (yt == 0)).sum() / neg
        fnr = ((yp == 0) & (yt == 1)).sum() / pos
        by_group[g] = {"sel": sel, "fpr": fpr, "fnr": fnr}

    sels = [v["sel"] for v in by_group.values()]
    fprs = [v["fpr"] for v in by_group.values()]
    fnrs = [v["fnr"] for v in by_group.values()]

    dp = float(np.nanmax(sels) - np.nanmin(sels))
    fpr_d = float(np.nanmax(fprs) - np.nanmin(fprs))
    fnr_d = float(np.nanmax(fnrs) - np.nanmin(fnrs))
    eo = max(fpr_d, fnr_d)

    return {
        "DP Diff": round(dp, 4),
        "EO Diff": round(eo, 4),
        "FPR Diff": round(fpr_d, 4),
        "FNR Diff": round(fnr_d, 4),
    }


def perf(y_true, y_pred):
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def make_bld(y, s):
    # AIF360 dataset for post-processing (labels + protected attr)
    return BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=pd.DataFrame({"label": y.astype(int), "sex": s.astype(int)}),
        label_names=["label"],
        protected_attribute_names=["sex"],
    )


def main():
    # 1) Load Adult
    ds = fetch_adult(as_frame=True)
    X = ds.data.copy()
    y = to_binary_income(ds.target)

    # Use sex as sensitive (Female=0, Male=1)
    s_raw = X["sex"].astype(str).values
    s = np.where(pd.Series(s_raw).str.lower().str.startswith("m"), 1, 0)

    # remove sensitive from model features (same fairness-audit style)
    X_model = X.drop(columns=["sex"], errors="ignore")

    # 2) Split train/val/test (val needed to fit EqOdds postprocessor)
    X_tr, X_tmp, y_tr, y_tmp, s_tr, s_tmp = train_test_split(
        X_model, y, s, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_te, y_val, y_te, s_val, s_te = train_test_split(
        X_tmp, y_tmp, s_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    # 3) Baseline model (logistic + preprocessing)
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_tr.select_dtypes(exclude=[np.number]).columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    clf = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, solver="saga", n_jobs=-1, random_state=42))])
    clf.fit(X_tr, y_tr)

    # Baseline predictions on test
    yhat_te_base = clf.predict(X_te).astype(int)

    # 4) AIF360 EqOdds postprocessing
    # Need predicted labels on validation and test
    yhat_val = clf.predict(X_val).astype(int)
    yhat_te = clf.predict(X_te).astype(int)

    val_true_ds = make_bld(y_val, s_val)
    val_pred_ds = make_bld(yhat_val, s_val)

    te_true_ds = make_bld(y_te, s_te)
    te_pred_ds = make_bld(yhat_te, s_te)

    cpp = EqOddsPostprocessing(
        unprivileged_groups=[{"sex": 0}],
        privileged_groups=[{"sex": 1}],
        seed=42
    )
    cpp = cpp.fit(val_true_ds, val_pred_ds)
    te_mit_ds = cpp.predict(te_pred_ds)
    yhat_te_mit = te_mit_ds.labels.ravel().astype(int)

    # 5) Metrics
    baseline_perf = perf(y_te, yhat_te_base)
    baseline_fair = fairness_diffs(y_te, yhat_te_base, s_te)

    mitigated_perf = perf(y_te, yhat_te_mit)
    mitigated_fair = fairness_diffs(y_te, yhat_te_mit, s_te)

    out = {
        "dataset": "adult",
        "sensitive": "sex",
        "constraint": "equalized_odds (AIF360 EqOddsPostprocessing)",
        "baseline": {"performance_test": baseline_perf, "fairness_test": baseline_fair},
        "aif360_mitigated": {"performance_test": mitigated_perf, "fairness_test": mitigated_fair},
    }

    print(json.dumps(out, indent=2))
    with open("benchmark_results/adult_sex_aif360_equalized_odds.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()