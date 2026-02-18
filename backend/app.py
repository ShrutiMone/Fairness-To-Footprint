# backend/app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import joblib
import os
import uuid
from datetime import datetime

from utils.fairness_metrics import (
    compute_fairness_metrics,
    generate_user_specific_suggestions,
    compute_performance_metrics,
    analyze_data_quality,
)
from utils.mitigation import (
    mitigate_with_exponentiated_gradient,
    mitigate_user_model,
    train_baseline_only,
    build_transformer,
    MitigatedBaselineWrapper,
    MitigatedUserModelWrapper,
)
from utils.model_loader import load_model
from threading import Thread
from typing import Optional
import time

# In-memory job/progress storage (simple, not persistent)
PROGRESS = {}  # job_id -> {status, percent, message}
RESULTS = {}   # job_id -> final result dict

app = Flask(__name__)
CORS(app)  # allow all origins for development

# Directory for storing models
MODEL_DIR = "saved_models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


# =============================================================================
# SHARED HELPER — runs the full analysis pipeline and returns a plain dict.
# Used by /analyze, /export_report, and the async worker for mitigate_async.
# Raises ValueError with a descriptive message on any expected failure so that
# callers can decide how to surface the error (jsonify vs PROGRESS update).
# =============================================================================

def _run_analysis(
    df: pd.DataFrame,
    target: str,
    sensitive: str,
    *,
    pred_col: Optional[str] = None,
    train_baseline_flag: bool = True,
    user_model_file=None,
    wrap_model_flag: bool = False,
    dp_threshold: float = 0.1,
    eo_threshold: float = 0.1,
    fpr_threshold: float = 0.1,
    fnr_threshold: float = 0.1,
) -> dict:
    """
    Core analysis logic shared across endpoints.

    Priority order for prediction source:
      1. User-uploaded model file (user_model_file)
      2. Train internal baseline   (train_baseline_flag=True)
      3. Existing prediction column (pred_col)

    Returns a serialisable dict ready to be jsonify()'d or written to a file.
    Raises ValueError for any expected/handled failure.
    """
    out = {}

    # ------------------------------------------------------------------
    # Branch 1: User uploaded a pre-trained model
    # ------------------------------------------------------------------
    if user_model_file is not None:
        try:
            user_model, is_dl_model = load_model(user_model_file, user_model_file.filename)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        except Exception as exc:
            raise ValueError(f"Failed to load model: {exc}") from exc

        # Guard: mitigated models are tied to a specific sensitive attribute
        if hasattr(user_model, "sensitive_col") and user_model.sensitive_col:
            if user_model.sensitive_col != sensitive:
                raise ValueError(
                    f"Uploaded mitigated model was created with sensitive column "
                    f"'{user_model.sensitive_col}'. Please use the same sensitive column."
                )

        # Predict — try native predict, fall back to wrap if requested
        wrapped = False
        try:
            if hasattr(user_model, "predict_with_sensitive"):
                y_pred = user_model.predict_with_sensitive(df, target_col=target, sensitive_col=sensitive)
            else:
                X = df.drop(columns=[target, sensitive], errors="ignore")
                y_pred = user_model.predict(X)
        except Exception as pred_exc:
            if wrap_model_flag and not hasattr(user_model, "predict_with_sensitive"):
                try:
                    from sklearn.pipeline import Pipeline as SKPipeline
                    transformer, _strat, _te = build_transformer(df, target, sensitive)
                    X = df.drop(columns=[target, sensitive], errors="ignore")
                    pipeline = SKPipeline([("pre", transformer), ("model", user_model)])
                    pipeline.fit(X, df[target])
                    y_pred = pipeline.predict(X)
                    wrapped = True
                except Exception as wrap_exc:
                    raise ValueError(
                        f"Failed to predict with or without wrapping: {wrap_exc}"
                    ) from wrap_exc
            else:
                raise ValueError(
                    f"Failed to run predict on uploaded model: {pred_exc}. "
                    f"You can enable 'wrap_model' to try applying standard preprocessing."
                ) from pred_exc

        tmp = df.copy()
        tmp["y_pred"] = y_pred
        metrics = compute_fairness_metrics(tmp, target, sensitive, pred_col="y_pred")
        suggestions = generate_user_specific_suggestions(
            df, metrics, target, sensitive,
            dp_threshold=dp_threshold, eo_threshold=eo_threshold,
            fpr_threshold=fpr_threshold, fnr_threshold=fnr_threshold,
        )
        out.update(metrics)
        out["suggestions"] = suggestions
        out["used_user_model"] = True
        out["wrapped_user_model"] = wrapped
        out["is_dl_model"] = is_dl_model
        out["performance"] = compute_performance_metrics(df[target], y_pred)
        out["data_quality"] = analyze_data_quality(df, target, sensitive)
        return out

    # ------------------------------------------------------------------
    # Branch 2: Train an internal baseline model
    # ------------------------------------------------------------------
    if train_baseline_flag:
        res = train_baseline_only(df, target, sensitive)
        res.pop("pipeline", None)

        metrics = res.get("metrics_baseline") or {}
        suggestions = generate_user_specific_suggestions(
            df, metrics, target, sensitive,
            dp_threshold=dp_threshold, eo_threshold=eo_threshold,
            fpr_threshold=fpr_threshold, fnr_threshold=fnr_threshold,
        )
        out.update(metrics)
        out["suggestions"] = suggestions
        out["strategy"] = res.get("strategy")
        out["time_estimate_seconds"] = res.get("time_estimate_seconds")
        out["is_dl_model"] = False  # Baseline is always sklearn
        out["performance_baseline"] = res.get("performance_baseline")
        out["performance_baseline_test"] = res.get("performance_baseline_test")
        out["data_quality"] = analyze_data_quality(df, target, sensitive)
        out["metrics_baseline_test"] = res.get("metrics_baseline_test")
        return out

    # ------------------------------------------------------------------
    # Branch 3: Use an existing prediction column in the CSV
    # ------------------------------------------------------------------
    if not pred_col:
        raise ValueError(
            "No prediction source provided. Upload a model, provide a prediction column, "
            "or enable 'Train baseline model internally'."
        )

    res = compute_fairness_metrics(df, target, sensitive, pred_col=pred_col)
    suggestions = generate_user_specific_suggestions(
        df, res, target, sensitive,
        dp_threshold=dp_threshold, eo_threshold=eo_threshold,
        fpr_threshold=fpr_threshold, fnr_threshold=fnr_threshold,
    )
    out.update(res)
    out["suggestions"] = suggestions
    out["is_dl_model"] = False
    out["performance"] = compute_performance_metrics(df[target], df[pred_col])
    out["data_quality"] = analyze_data_quality(df, target, sensitive)
    return out


# =============================================================================
# SHARED HELPER — parse common form fields & file from a Flask request.
# Returns (df, target, sensitive, kwargs_for_run_analysis) or raises ValueError.
# =============================================================================

def _parse_analysis_request(req):
    """
    Parse the common fields used by /analyze and /export_report.
    Returns (df, target, sensitive, analysis_kwargs).
    Raises ValueError on missing/invalid inputs.
    """
    if "file" not in req.files:
        raise ValueError("No file uploaded")

    file = req.files["file"]
    target = req.form.get("target")
    sensitive = req.form.get("sensitive")

    if not target or not sensitive:
        raise ValueError("target and sensitive are required")

    df = pd.read_csv(file)

    kwargs = dict(
        pred_col=req.form.get("pred_col") or None,
        train_baseline_flag=req.form.get("train_baseline", "true").lower() in ("1", "true", "yes"),
        user_model_file=req.files.get("user_model") or None,
        wrap_model_flag=req.form.get("wrap_model", "false").lower() in ("1", "true", "yes"),
        dp_threshold=float(req.form.get("dp_threshold", 0.1)),
        eo_threshold=float(req.form.get("eo_threshold", 0.1)),
        fpr_threshold=float(req.form.get("fpr_threshold", 0.1)),
        fnr_threshold=float(req.form.get("fnr_threshold", 0.1)),
    )

    return df, target, sensitive, kwargs


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        df, target, sensitive, kwargs = _parse_analysis_request(request)
        out = _run_analysis(df, target, sensitive, **kwargs)
        return jsonify(out)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/export_report", methods=["POST"])
def export_report():
    """
    Generate a downloadable JSON report for audit/compliance.
    Uses the same inputs as /analyze.
    """
    import tempfile
    import json

    try:
        df, target, sensitive, kwargs = _parse_analysis_request(request)
        out = _run_analysis(df, target, sensitive, **kwargs)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmpf:
            json.dump(out, tmpf, indent=2)
            tmp_path = tmpf.name

        return send_file(tmp_path, as_attachment=True, download_name="fairness_report.json")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/mitigate_async", methods=["POST"])
def mitigate_async():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        target = request.form.get("target")
        sensitive = request.form.get("sensitive")
        constraint = request.form.get("constraint", "demographic_parity")

        if not target or not sensitive:
            return jsonify({"error": "target and sensitive are required"}), 400

        df = pd.read_csv(file)
        job_id = str(uuid.uuid4())
        PROGRESS[job_id] = {"status": "running", "percent": 0, "message": "queued"}

        def worker(df, target, sensitive, constraint, job_id):
            try:
                PROGRESS[job_id].update({"percent": 10, "message": "starting mitigation"})
                res = mitigate_with_exponentiated_gradient(df, target, sensitive, constraint=constraint)

                mitigator = res.pop("mitigator")
                transformer = res.pop("transformer")
                label_encoder = res.pop("label_encoder")

                model_id = str(uuid.uuid4())
                model_path = os.path.join(MODEL_DIR, f"{model_id}.joblib")
                wrapper = MitigatedBaselineWrapper(
                    mitigator=mitigator,
                    target_col=target,
                    sensitive_col=sensitive,
                    metadata={
                        "transformer": transformer,
                        "label_encoder": label_encoder,
                        "constraint": constraint,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                joblib.dump(wrapper, model_path)
                res["model_id"] = model_id
                res["model_download_url"] = f"http://127.0.0.1:5000/download_model/{model_id}"

                PROGRESS[job_id].update({"percent": 100, "message": "done", "status": "done"})
                RESULTS[job_id] = res
            except Exception as exc:
                PROGRESS[job_id].update({"status": "failed", "message": str(exc)})
                RESULTS[job_id] = {"error": str(exc)}

        t = Thread(target=worker, args=(df, target, sensitive, constraint, job_id), daemon=True)
        t.start()

        return jsonify({"job_id": job_id}), 202
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/mitigate_user_model_async", methods=["POST"])
def mitigate_user_model_async():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No data file uploaded"}), 400
        if "user_model" not in request.files:
            return jsonify({"error": "No user model file uploaded"}), 400

        data_file = request.files["file"]
        model_file = request.files["user_model"]
        target = request.form.get("target")
        sensitive = request.form.get("sensitive")
        constraint = request.form.get("constraint", "demographic_parity")
        wrap_model_flag = request.form.get("wrap_model", "false").lower() in ("1", "true", "yes")

        if not target or not sensitive:
            return jsonify({"error": "target and sensitive are required"}), 400

        df = pd.read_csv(data_file)

        try:
            user_model, is_dl_model = load_model(model_file, model_file.filename)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"Failed to load model: {exc}"}), 400

        if is_dl_model:
            return jsonify({"error": "Mitigation is not supported for deep-learning models."}), 400

        job_id = str(uuid.uuid4())
        PROGRESS[job_id] = {"status": "running", "percent": 0, "message": "queued"}

        def worker(df, user_model, target, sensitive, constraint, job_id, wrap_model_flag):
            try:
                PROGRESS[job_id].update({"percent": 5, "message": "starting"})
                time.sleep(0.1)

                PROGRESS[job_id].update({"percent": 20, "message": "computing baseline predictions"})
                res = mitigate_user_model(df, user_model, target, sensitive, constraint=constraint)

                PROGRESS[job_id].update({"percent": 90, "message": "finalizing results"})
                uploaded_model = res.pop("user_model", None)
                final_model = res.pop("final_model", None)
                transformer = res.pop("transformer", None)
                group_thresholds = res.get("group_thresholds", {})

                model_id = str(uuid.uuid4())
                model_path = os.path.join(MODEL_DIR, f"{model_id}.joblib")
                wrapper = MitigatedUserModelWrapper(
                    final_model=final_model,
                    transformer=transformer,
                    group_thresholds=group_thresholds,
                    sensitive_col=sensitive,
                    target_col=target,
                    default_threshold=0.5,
                    constraint=constraint,
                    metadata={
                        "original_model": uploaded_model,
                        "mitigation_type": res.get("mitigation_type"),
                    },
                )
                joblib.dump(wrapper, model_path)
                res["model_id"] = model_id
                res["model_download_url"] = f"http://127.0.0.1:5000/download_model/{model_id}"

                PROGRESS[job_id].update({"percent": 100, "message": "done", "status": "done"})
                RESULTS[job_id] = res
            except Exception as exc:
                PROGRESS[job_id].update({"status": "failed", "message": str(exc)})
                RESULTS[job_id] = {"error": str(exc)}

        t = Thread(
            target=worker,
            args=(df, user_model, target, sensitive, constraint, job_id, wrap_model_flag),
            daemon=True,
        )
        t.start()

        return jsonify({"job_id": job_id}), 202
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/progress/<job_id>", methods=["GET"])
def get_progress(job_id):
    return jsonify(PROGRESS.get(job_id, {"status": "unknown", "percent": 0, "message": "no job"}))


@app.route("/result/<job_id>", methods=["GET"])
def get_result(job_id):
    return jsonify(RESULTS.get(job_id, {"error": "result not ready"}))


@app.route("/download_model/<model_id>", methods=["GET"])
def download_model(model_id):
    try:
        model_path = os.path.join(MODEL_DIR, f"{model_id}.joblib")
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not found"}), 404
        return send_file(model_path, as_attachment=True, download_name=f"mitigated_model_{model_id}.joblib")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)