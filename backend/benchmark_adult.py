"""Run fairness benchmark on Fairlearn Adult dataset for Results & Discussion section.

Usage:
  cd backend
  python benchmark_adult.py --sensitive sex --constraint demographic_parity
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from fairlearn.datasets import fetch_adult

from utils.mitigation import train_baseline_only, mitigate_with_exponentiated_gradient


def _to_binary_target(y: pd.Series) -> pd.Series:
    ys = y.astype(str).str.strip().str.lower()
    positive = ys.str.contains('>50k') | ys.eq('1') | ys.eq('true')
    return positive.astype(int)


def _first_present(d: dict, *keys):
    for k in keys:
        if k in d and d.get(k) is not None:
            return d.get(k)
    # If key exists but value is None, return None for first existing key.
    for k in keys:
        if k in d:
            return d.get(k)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitive", default="sex", help="Sensitive attribute column: sex or race")
    parser.add_argument("--constraint", default="demographic_parity", choices=["demographic_parity", "equalized_odds"])
    parser.add_argument("--output_dir", default="benchmark_results")
    parser.add_argument("--csv_path", default="", help="Optional local Adult CSV path for offline runs")
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    log(f"[benchmark] Starting Adult fairness benchmark (sensitive={args.sensitive}, constraint={args.constraint})")
    log(f"[benchmark] Output directory: {out_dir}")

    if args.csv_path:
        log(f"[benchmark] Loading local CSV: {args.csv_path}")
        df = pd.read_csv(args.csv_path)
        if "income" not in df.columns:
            raise ValueError("Local CSV must include an 'income' target column.")
        df["income"] = _to_binary_target(df["income"])
    else:
        try:
            log("[benchmark] Downloading Adult dataset via fairlearn.datasets.fetch_adult ...")
            ds = fetch_adult(as_frame=True)
            X = ds.data.copy()
            y = _to_binary_target(pd.Series(ds.target, name="income"))
            if args.sensitive not in X.columns:
                raise ValueError(f"Sensitive column '{args.sensitive}' not in dataset columns: {list(X.columns)}")
            df = X.copy()
            df["income"] = y
            log(f"[benchmark] Loaded dataset shape: {df.shape}")
        except Exception as exc:
            message = (
                "Failed to download Adult dataset via fetch_adult. "
                "Use --csv_path <local_adult_csv> for offline/proxy-restricted environments."
            )
            (out_dir / "last_error.txt").write_text(f"{message}\n\nOriginal error:\n{exc}\n")
            raise RuntimeError(message) from exc

    log("[benchmark] Training baseline model ...")
    baseline = train_baseline_only(df, target_col="income", sensitive_col=args.sensitive)
    log("[benchmark] Running mitigation (ExponentiatedGradient) ...")
    mitigated = mitigate_with_exponentiated_gradient(
        df,
        target_col="income",
        sensitive_col=args.sensitive,
        constraint=args.constraint,
    )

    mitigated_perf_test = _first_present(mitigated, "performance_after_mitigation_test", "performance_after_test")
    mitigated_metrics_test = _first_present(mitigated, "metrics_after_mitigation_test", "metrics_after_test")

    result = {
        "dataset": "fairlearn.datasets.fetch_adult",
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target": "income",
        "sensitive": args.sensitive,
        "constraint": args.constraint,
        "baseline": {
            "performance_test": baseline.get("performance_baseline_test"),
            "fairness_test": baseline.get("metrics_baseline_test", {}).get("overall", {}),
        },
        "mitigated": {
            "performance_test": mitigated_perf_test,
            "fairness_test": (mitigated_metrics_test or {}).get("overall", {}),
        },
        "debug": {
            "baseline_metrics_test_raw": baseline.get("metrics_baseline_test"),
            "mitigated_metrics_test_raw": mitigated_metrics_test,
            "mitigation_result_keys": sorted(list(mitigated.keys())),
        },
    }

    out_json = out_dir / f"adult_{args.sensitive}_{args.constraint}.json"
    out_json.write_text(json.dumps(result, indent=2))

    md = [
        "# Adult Benchmark Summary",
        "",
        f"- Dataset: `{result['dataset']}`",
        f"- Rows: {result['rows']}",
        f"- Sensitive attribute: `{args.sensitive}`",
        f"- Constraint: `{args.constraint}`",
        "",
        "## Holdout Performance",
        "",
        f"- Baseline: {result['baseline']['performance_test']}",
        f"- Mitigated: {result['mitigated']['performance_test']}",
        "",
        "## Holdout Fairness (overall)",
        "",
        f"- Baseline: {result['baseline']['fairness_test']}",
        f"- Mitigated: {result['mitigated']['fairness_test']}",
        "",
        "## Discussion starter",
        "",
        "Mitigation is successful when fairness disparities (e.g., DP difference / EO difference) drop meaningfully on holdout data while utility metrics (accuracy/F1) remain acceptable for the intended use case.",
    ]
    out_md = out_dir / f"adult_{args.sensitive}_{args.constraint}.md"
    out_md.write_text("\n".join(md))

    log(f"[benchmark] Saved: {out_json}")
    log(f"[benchmark] Saved: {out_md}")
    log("[benchmark] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[benchmark] ERROR: {exc}", file=sys.stderr, flush=True)
        raise
