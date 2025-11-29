# backend/app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import joblib
import os
import uuid
from datetime import datetime

from utils.fairness_metrics import compute_fairness_metrics, generate_user_specific_suggestions
from utils.mitigation import mitigate_with_exponentiated_gradient, mitigate_user_model, train_baseline_only, build_transformer
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
        # Optional behavior: train a baseline model internally or use an uploaded user model
        train_baseline_flag = request.form.get("train_baseline", "true").lower() in ("1", "true", "yes")
        user_model_file = request.files.get("user_model", None)

        if not target or not sensitive:
            return jsonify({"error": "target and sensitive are required"}), 400

        df = pd.read_csv(file)

        # If a user-uploaded model is present, use it to produce predictions
        if user_model_file:
            wrap_model_flag = request.form.get("wrap_model", "false").lower() in ("1", "true", "yes")
            try:
                user_model = joblib.load(user_model_file)
            except Exception:
                import pickle
                user_model_file.seek(0)
                user_model = pickle.load(user_model_file)

            # Prepare feature matrix for prediction (drop target and sensitive)
            X = df.drop(columns=[target, sensitive], errors='ignore')
            # First try to predict as-is
            try:
                y_pred = user_model.predict(X)
                wrapped = False
            except Exception as e:
                # If prediction fails and user asked for wrapping, try to build transformer and wrap
                if wrap_model_flag:
                    try:
                        transformer, strat, te = build_transformer(df, target, sensitive)
                        from sklearn.pipeline import Pipeline as SKPipeline
                        pipeline = SKPipeline([("pre", transformer), ("model", user_model)])
                        y_pred = pipeline.predict(X)
                        wrapped = True
                    except Exception as e2:
                        return jsonify({"error": f"Failed to predict with or without wrapping: {str(e2)}"}), 400
                else:
                    return jsonify({"error": f"Failed to run predict on uploaded model: {str(e)}. You can enable 'wrap_model' to try applying standard preprocessing."}), 400

            tmp = df.copy()
            tmp["y_pred"] = y_pred
            metrics = compute_fairness_metrics(tmp, target, sensitive, pred_col="y_pred")
            suggestions = generate_user_specific_suggestions(df, metrics, target, sensitive)
            # Return in same shape as previous API: top-level overall/by_group keys
            out = {}
            out.update(metrics)
            out["suggestions"] = suggestions
            out["used_user_model"] = True
            out["wrapped_user_model"] = bool(wrapped)
            return jsonify(out)

        # If requested, train a baseline model internally and produce predictions+metrics
        if train_baseline_flag:
            # Use a lightweight baseline-only trainer (faster) to produce baseline predictions and metrics.
            res = train_baseline_only(df, target, sensitive)

            # Remove heavy objects before returning
            res.pop("pipeline", None)

            metrics = res.get("metrics_baseline") or {}
            suggestions = generate_user_specific_suggestions(df, metrics, target, sensitive)
            out = {}
            out.update(metrics)
            out["suggestions"] = suggestions
            out["strategy"] = res.get("strategy")
            out["time_estimate_seconds"] = res.get("time_estimate_seconds")
            return jsonify(out)

        # Default: compute metrics using provided pred_col (or default to y_true if no pred_col)
        res = compute_fairness_metrics(df, target, sensitive, pred_col=pred_col)
        suggestions = generate_user_specific_suggestions(df, res, target, sensitive)
        out = {}
        out.update(res)
        out["suggestions"] = suggestions
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
        
        # Extract non-serializable objects
        mitigator = result.pop("mitigator")
        transformer = result.pop("transformer")
        label_encoder = result.pop("label_encoder")
        
        # Save model to disk
        model_id = str(uuid.uuid4())
        model_path = os.path.join(MODEL_DIR, f"{model_id}.joblib")
        model_metadata = {
            "mitigator": mitigator,
            "transformer": transformer,
            "label_encoder": label_encoder,
            "target_col": target,
            "sensitive_col": sensitive,
            "constraint": constraint,
            "timestamp": datetime.now().isoformat()
        }
        joblib.dump(model_metadata, model_path)
        
        # Add download link to response (with full URL)
        result["model_id"] = model_id
        result["model_download_url"] = f"http://127.0.0.1:5000/download_model/{model_id}"
        
        return jsonify(result)
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
                model_metadata = {
                    "mitigator": mitigator,
                    "transformer": transformer,
                    "label_encoder": label_encoder,
                    "target_col": target,
                    "sensitive_col": sensitive,
                    "constraint": constraint
                }
                joblib.dump(model_metadata, model_path)
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

        if not target or not sensitive:
            return jsonify({"error": "target and sensitive are required"}), 400

        df = pd.read_csv(data_file)

        # load model
        try:
            user_model = joblib.load(model_file)
        except Exception:
            import pickle
            model_file.seek(0)
            user_model = pickle.load(model_file)

        job_id = str(uuid.uuid4())
        PROGRESS[job_id] = {"status": "running", "percent": 0, "message": "queued"}

        def worker(df, user_model, target, sensitive, constraint, job_id):
            try:
                PROGRESS[job_id].update({"percent": 5, "message": "starting"})
                time.sleep(0.1)
                PROGRESS[job_id].update({"percent": 20, "message": "computing baseline predictions"})
                res = mitigate_user_model(df, user_model, target, sensitive, constraint=constraint)

                PROGRESS[job_id].update({"percent": 90, "message": "finalizing results"})
                mitigator = res.pop("mitigator", None)
                uploaded_model = res.pop("user_model", None)

                model_id = str(uuid.uuid4())
                model_path = os.path.join(MODEL_DIR, f"{model_id}.joblib")
                model_metadata = {
                    "mitigator": mitigator,
                    "original_model": uploaded_model,
                    "target_col": target,
                    "sensitive_col": sensitive,
                    "constraint": constraint
                }
                joblib.dump(model_metadata, model_path)
                res["model_id"] = model_id
                res["model_download_url"] = f"http://127.0.0.1:5000/download_model/{model_id}"

                PROGRESS[job_id].update({"percent": 100, "message": "done", "status": "done"})
                RESULTS[job_id] = res
            except Exception as e:
                PROGRESS[job_id].update({"status": "failed", "message": str(e)})
                RESULTS[job_id] = {"error": str(e)}

        t = Thread(target=worker, args=(df, user_model, target, sensitive, constraint, job_id), daemon=True)
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
    - user_model: Pre-trained model (.joblib file)
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
        
        if not target or not sensitive:
            return jsonify({"error": "target and sensitive are required"}), 400
        
        # Load data
        df = pd.read_csv(data_file)
        
        # Load user model
        try:
            # If it's a .joblib file
            user_model = joblib.load(model_file)
        except Exception:
            # If it's a .pkl file
            import pickle
            model_file.seek(0)
            user_model = pickle.load(model_file)
        
        # Apply mitigation
        result = mitigate_user_model(df, user_model, target, sensitive, constraint=constraint)
        
        # Extract non-serializable objects
        mitigator = result.pop("mitigator")
        uploaded_model = result.pop("user_model")
        
        # Save mitigated model to disk
        model_id = str(uuid.uuid4())
        model_path = os.path.join(MODEL_DIR, f"{model_id}.joblib")
        model_metadata = {
            "mitigator": mitigator,
            "original_model": uploaded_model,
            "target_col": target,
            "sensitive_col": sensitive,
            "constraint": constraint,
            "timestamp": datetime.now().isoformat()
        }
        joblib.dump(model_metadata, model_path)
        
        # Add download link to response
        result["model_id"] = model_id
        result["model_download_url"] = f"http://127.0.0.1:5000/download_model/{model_id}"
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
