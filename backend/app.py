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
import uuid
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

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        target = request.form.get("target")
        sensitive = request.form.get("sensitive")
        pred_col = request.form.get("pred_col", None)
        dp_threshold = float(request.form.get("dp_threshold", 0.1))
        eo_threshold = float(request.form.get("eo_threshold", 0.1))
        fpr_threshold = float(request.form.get("fpr_threshold", 0.1))
        fnr_threshold = float(request.form.get("fnr_threshold", 0.1))
        # Optional behavior: train a baseline model internally or use an uploaded user model
        train_baseline_flag = request.form.get("train_baseline", "true").lower() in ("1", "true", "yes")
        user_model_file = request.files.get("user_model", None)

        if not target or not sensitive:
            return jsonify({"error": "target and sensitive are required"}), 400

        df = pd.read_csv(file)

        # If a user-uploaded model is present, use it to produce predictions
        if user_model_file:
            wrap_model_flag = request.form.get("wrap_model", "false").lower() in ("1", "true", "yes")
            
            # Load model using new model loader (handles joblib, ONNX, Keras, PyTorch, etc.)
            try:
                user_model, is_dl_model = load_model(user_model_file, user_model_file.filename)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                return jsonify({"error": f"Failed to load model: {str(e)}"}), 400

            # Guard: mitigated models are tied to a specific sensitive attribute
            if hasattr(user_model, "sensitive_col") and user_model.sensitive_col:
                if user_model.sensitive_col != sensitive:
                    return jsonify({"error": f"Uploaded mitigated model was created with sensitive column '{user_model.sensitive_col}'. Please use the same sensitive column."}), 400

            # Predict: if model requires sensitive column, pass full df
            try:
                if hasattr(user_model, "predict_with_sensitive"):
                    y_pred = user_model.predict_with_sensitive(df, target_col=target, sensitive_col=sensitive)
                    wrapped = False
                else:
                    X = df.drop(columns=[target, sensitive], errors='ignore')
                    y_pred = user_model.predict(X)
                    wrapped = False
            except Exception as e:
                # If prediction fails and user asked for wrapping, try to build transformer and wrap
                if wrap_model_flag and not hasattr(user_model, "predict_with_sensitive"):
                    try:
                        transformer, strat, te = build_transformer(df, target, sensitive)
                        from sklearn.pipeline import Pipeline as SKPipeline
                        X = df.drop(columns=[target, sensitive], errors='ignore')
                        pipeline = SKPipeline([("pre", transformer), ("model", user_model)])
                        # Fit the pipeline's preprocessing step on the full data
                        pipeline.fit(X, df[target])
                        y_pred = pipeline.predict(X)
                        wrapped = True
                    except Exception as e2:
                        return jsonify({"error": f"Failed to predict with or without wrapping: {str(e2)}"}), 400
                else:
                    return jsonify({"error": f"Failed to run predict on uploaded model: {str(e)}. You can enable 'wrap_model' to try applying standard preprocessing."}), 400

            tmp = df.copy()
            tmp["y_pred"] = y_pred
            metrics = compute_fairness_metrics(tmp, target, sensitive, pred_col="y_pred")
            performance = compute_performance_metrics(df[target], y_pred)
            data_quality = analyze_data_quality(df, target, sensitive)
            suggestions = generate_user_specific_suggestions(
                df, metrics, target, sensitive,
                dp_threshold=dp_threshold, eo_threshold=eo_threshold,
                fpr_threshold=fpr_threshold, fnr_threshold=fnr_threshold
            )
            # Return in same shape as previous API: top-level overall/by_group keys
            out = {}
            out.update(metrics)
            out["suggestions"] = suggestions
            out["used_user_model"] = True
            out["wrapped_user_model"] = bool(wrapped)
            out["is_dl_model"] = is_dl_model  # New flag: True if model is deep learning
            out["performance"] = performance
            out["data_quality"] = data_quality
            return jsonify(out)

        # If requested, train a baseline model internally and produce predictions+metrics
        if train_baseline_flag:
            # Use a lightweight baseline-only trainer (faster) to produce baseline predictions and metrics.
            res = train_baseline_only(df, target, sensitive)

            # Remove heavy objects before returning
            res.pop("pipeline", None)

            metrics = res.get("metrics_baseline") or {}
            suggestions = generate_user_specific_suggestions(
                df, metrics, target, sensitive,
                dp_threshold=dp_threshold, eo_threshold=eo_threshold,
                fpr_threshold=fpr_threshold, fnr_threshold=fnr_threshold
            )
            out = {}
            out.update(metrics)
            out["suggestions"] = suggestions
            out["strategy"] = res.get("strategy")
            out["time_estimate_seconds"] = res.get("time_estimate_seconds")
            out["is_dl_model"] = False  # Baseline is always sklearn
            out["performance_baseline"] = res.get("performance_baseline")
            out["performance_baseline_test"] = res.get("performance_baseline_test")
            out["data_quality"] = analyze_data_quality(df, target, sensitive)
            out["metrics_baseline_test"] = res.get("metrics_baseline_test")
            return jsonify(out)

        # Default: compute metrics using provided pred_col
        if not pred_col:
            return jsonify({"error": "No prediction source provided. Upload a model, provide a prediction column, or enable 'Train baseline model internally'."}), 400
        res = compute_fairness_metrics(df, target, sensitive, pred_col=pred_col)
        suggestions = generate_user_specific_suggestions(
            df, res, target, sensitive,
            dp_threshold=dp_threshold, eo_threshold=eo_threshold,
            fpr_threshold=fpr_threshold, fnr_threshold=fnr_threshold
        )
        out = {}
        out.update(res)
        out["suggestions"] = suggestions
        out["is_dl_model"] = False  # Default case is sklearn
        out["performance"] = compute_performance_metrics(df[target], df[pred_col] if pred_col else df[target])
        out["data_quality"] = analyze_data_quality(df, target, sensitive)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/mitigate", methods=["POST"])
def mitigate():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        target = request.form.get("target")
        sensitive = request.form.get("sensitive")
        constraint = request.form.get("constraint", "demographic_parity")  # or "equalized_odds"

        if not target or not sensitive:
            return jsonify({"error": "target and sensitive are required"}), 400

        df = pd.read_csv(file)
        result = mitigate_with_exponentiated_gradient(df, target, sensitive, constraint=constraint)
        
        # Extract non-serializable objects for wrapper creation
        mitigator = result.pop("mitigator")
        transformer = result.pop("transformer")
        label_encoder = result.pop("label_encoder")
        
        # Save mitigated model wrapper to disk
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
        
        # Add download link to response (with full URL)
        result["model_id"] = model_id
        result["model_download_url"] = f"http://127.0.0.1:5000/download_model/{model_id}"
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/export_report", methods=["POST"])
def export_report():
    """
    Generate a downloadable JSON report for audit/compliance.
    Uses the same inputs as /analyze.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        target = request.form.get("target")
        sensitive = request.form.get("sensitive")
        pred_col = request.form.get("pred_col", None)
        train_baseline_flag = request.form.get("train_baseline", "true").lower() in ("1", "true", "yes")
        user_model_file = request.files.get("user_model", None)
        wrap_model_flag = request.form.get("wrap_model", "false").lower() in ("1", "true", "yes")
        dp_threshold = float(request.form.get("dp_threshold", 0.1))
        eo_threshold = float(request.form.get("eo_threshold", 0.1))
        fpr_threshold = float(request.form.get("fpr_threshold", 0.1))
        fnr_threshold = float(request.form.get("fnr_threshold", 0.1))

        if not target or not sensitive:
            return jsonify({"error": "target and sensitive are required"}), 400

        # Reuse analyze logic by calling /analyze-like flow locally
        df = pd.read_csv(file)
        out = {}

        if user_model_file:
            try:
                user_model, is_dl_model = load_model(user_model_file, user_model_file.filename)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                return jsonify({"error": f"Failed to load model: {str(e)}"}), 400

            if hasattr(user_model, "sensitive_col") and user_model.sensitive_col:
                if user_model.sensitive_col != sensitive:
                    return jsonify({"error": f"Uploaded mitigated model was created with sensitive column '{user_model.sensitive_col}'. Please use the same sensitive column."}), 400

            # Predict
            try:
                if hasattr(user_model, "predict_with_sensitive"):
                    y_pred = user_model.predict_with_sensitive(df, target_col=target, sensitive_col=sensitive)
                    wrapped = False
                else:
                    X = df.drop(columns=[target, sensitive], errors='ignore')
                    y_pred = user_model.predict(X)
                    wrapped = False
            except Exception as e:
                if wrap_model_flag and not hasattr(user_model, "predict_with_sensitive"):
                    try:
                        transformer, strat, te = build_transformer(df, target, sensitive)
                        from sklearn.pipeline import Pipeline as SKPipeline
                        X = df.drop(columns=[target, sensitive], errors='ignore')
                        pipeline = SKPipeline([("pre", transformer), ("model", user_model)])
                        pipeline.fit(X, df[target])
                        y_pred = pipeline.predict(X)
                        wrapped = True
                    except Exception as e2:
                        return jsonify({"error": f"Failed to predict with or without wrapping: {str(e2)}"}), 400
                else:
                    return jsonify({"error": f"Failed to run predict on uploaded model: {str(e)}. You can enable 'wrap_model' to try applying standard preprocessing."}), 400

            tmp = df.copy()
            tmp["y_pred"] = y_pred
            metrics = compute_fairness_metrics(tmp, target, sensitive, pred_col="y_pred")
            suggestions = generate_user_specific_suggestions(
                df, metrics, target, sensitive,
                dp_threshold=dp_threshold, eo_threshold=eo_threshold,
                fpr_threshold=fpr_threshold, fnr_threshold=fnr_threshold
            )
            out.update(metrics)
            out["suggestions"] = suggestions
            out["used_user_model"] = True
            out["wrapped_user_model"] = bool(wrapped)
            out["is_dl_model"] = is_dl_model
            out["performance"] = compute_performance_metrics(df[target], y_pred)
            out["data_quality"] = analyze_data_quality(df, target, sensitive)
        elif train_baseline_flag:
            res = train_baseline_only(df, target, sensitive)
            res.pop("pipeline", None)
            metrics = res.get("metrics_baseline") or {}
            suggestions = generate_user_specific_suggestions(
                df, metrics, target, sensitive,
                dp_threshold=dp_threshold, eo_threshold=eo_threshold,
                fpr_threshold=fpr_threshold, fnr_threshold=fnr_threshold
            )
            out.update(metrics)
            out["suggestions"] = suggestions
            out["strategy"] = res.get("strategy")
            out["time_estimate_seconds"] = res.get("time_estimate_seconds")
            out["is_dl_model"] = False
            out["performance_baseline"] = res.get("performance_baseline")
            out["performance_baseline_test"] = res.get("performance_baseline_test")
            out["data_quality"] = analyze_data_quality(df, target, sensitive)
            out["metrics_baseline_test"] = res.get("metrics_baseline_test")
        else:
            res = compute_fairness_metrics(df, target, sensitive, pred_col=pred_col)
            suggestions = generate_user_specific_suggestions(
                df, res, target, sensitive,
                dp_threshold=dp_threshold, eo_threshold=eo_threshold,
                fpr_threshold=fpr_threshold, fnr_threshold=fnr_threshold
            )
            out.update(res)
            out["suggestions"] = suggestions
            out["is_dl_model"] = False
            out["performance"] = compute_performance_metrics(df[target], df[pred_col] if pred_col else df[target])
            out["data_quality"] = analyze_data_quality(df, target, sensitive)

        # Save to temp file and return as download
        import tempfile
        import json
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmpf:
            json.dump(out, tmpf, indent=2)
            tmp_path = tmpf.name
        return send_file(tmp_path, as_attachment=True, download_name="fairness_report.json")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/mitigate_async", methods=["POST"])
def mitigate_async():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        target = request.form.get("target")
        sensitive = request.form.get("sensitive")
        constraint = request.form.get("constraint", "demographic_parity")  # or "equalized_odds"

        if not target or not sensitive:
            return jsonify({"error": "target and sensitive are required"}), 400

        # Read CSV into DataFrame now (small cost) and start background job
        df = pd.read_csv(file)

        job_id = str(uuid.uuid4())
        PROGRESS[job_id] = {"status": "running", "percent": 0, "message": "queued"}

        def worker(df, target, sensitive, constraint, job_id):
            try:
                PROGRESS[job_id].update({"percent": 2, "message": "starting"})
                time.sleep(0.1)

                PROGRESS[job_id].update({"percent": 8, "message": "preprocessing features"})
                time.sleep(0.1)

                # Stage: train baseline pipeline
                PROGRESS[job_id].update({"percent": 20, "message": "training baseline model"})
                res = mitigate_with_exponentiated_gradient(df, target, sensitive, constraint=constraint)

                PROGRESS[job_id].update({"percent": 85, "message": "computing mitigated model"})
                # Save model metadata and make result serializable (similar to /mitigate)
                mitigator = res.pop("mitigator", None)
                transformer = res.pop("transformer", None)
                label_encoder = res.pop("label_encoder", None)

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
                    },
                )
                joblib.dump(wrapper, model_path)
                res["model_id"] = model_id
                res["model_download_url"] = f"http://127.0.0.1:5000/download_model/{model_id}"

                PROGRESS[job_id].update({"percent": 100, "message": "done", "status": "done"})
                RESULTS[job_id] = res
            except Exception as e:
                PROGRESS[job_id].update({"status": "failed", "message": str(e)})
                RESULTS[job_id] = {"error": str(e)}

        t = Thread(target=worker, args=(df, target, sensitive, constraint, job_id), daemon=True)
        t.start()

        return jsonify({"job_id": job_id}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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

        # Load model using model loader
        try:
            user_model, is_dl_model = load_model(model_file, model_file.filename)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 400

        # Check if DL model
        if is_dl_model:
            return jsonify({"error": "Mitigation is not supported for deep-learning models."}), 400

        job_id = str(uuid.uuid4())
        PROGRESS[job_id] = {"status": "running", "percent": 0, "message": "queued"}

        def worker(df, user_model, target, sensitive, constraint, job_id, wrap_model_flag):
            try:
                PROGRESS[job_id].update({"percent": 5, "message": "starting"})
                time.sleep(0.1)
                
                # Prepare features
                X = df.drop(columns=[target, sensitive], errors='ignore')
                
                # Use model as-is - mitigate_user_model will handle Pipeline extraction
                user_model_for_mitigation = user_model
                
                PROGRESS[job_id].update({"percent": 20, "message": "computing baseline predictions"})
                res = mitigate_user_model(df, user_model_for_mitigation, target, sensitive, constraint=constraint)

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
            except Exception as e:
                PROGRESS[job_id].update({"status": "failed", "message": str(e)})
                RESULTS[job_id] = {"error": str(e)}

        t = Thread(target=worker, args=(df, user_model, target, sensitive, constraint, job_id, wrap_model_flag), daemon=True)
        t.start()

        return jsonify({"job_id": job_id}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/progress/<job_id>', methods=['GET'])
def get_progress(job_id):
    return jsonify(PROGRESS.get(job_id, {"status": "unknown", "percent": 0, "message": "no job"}))


@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id):
    return jsonify(RESULTS.get(job_id, {"error": "result not ready"}))

@app.route("/download_model/<model_id>", methods=["GET"])
def download_model(model_id):
    try:
        model_path = os.path.join(MODEL_DIR, f"{model_id}.joblib")
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not found"}), 404
        return send_file(model_path, as_attachment=True, download_name=f"mitigated_model_{model_id}.joblib")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/mitigate_user_model", methods=["POST"])
def mitigate_uploaded_model():
    """
    Endpoint to apply fairness mitigation to a user-provided pre-trained model.
    
    Expects:
    - file: CSV data with target and sensitive columns
    - user_model: Pre-trained model (.joblib, .pkl, .onnx, .keras, .pt, .pth, etc.)
    - target: Name of target column
    - sensitive: Name of sensitive attribute column
    - constraint: "demographic_parity" or "equalized_odds"
    """
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
        
        # Load data
        df = pd.read_csv(data_file)
        
        # Load user model using the model loader
        try:
            user_model, is_dl_model = load_model(model_file, model_file.filename)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 400
        
        # DL models don't support mitigation, only analysis
        if is_dl_model:
            return jsonify({"error": "Mitigation is not supported for deep-learning models. Use analysis to view fairness metrics only."}), 400
        
        # Prepare feature matrix
        X = df.drop(columns=[target, sensitive], errors='ignore')
        
        # Use model as-is - mitigate_user_model will handle Pipeline extraction
        user_model_for_mitigation = user_model
        
        # Apply mitigation
        try:
            result = mitigate_user_model(df, user_model_for_mitigation, target, sensitive, constraint=constraint)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"Mitigation failed: {str(e)}"}), 400
        
        # Extract non-serializable objects
        uploaded_model = result.pop("user_model")
        final_model = result.pop("final_model", None)
        transformer = result.pop("transformer", None)
        group_thresholds = result.get("group_thresholds", {})
        
        # Save mitigated model to disk
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
                "mitigation_type": result.get("mitigation_type"),
                "timestamp": datetime.now().isoformat(),
            },
        )
        joblib.dump(wrapper, model_path)
        
        # Add download link to response
        result["model_id"] = model_id
        result["model_download_url"] = f"http://127.0.0.1:5000/download_model/{model_id}"
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
