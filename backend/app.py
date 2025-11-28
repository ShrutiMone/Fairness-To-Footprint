# backend/app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import joblib
import os
import uuid
from datetime import datetime

from utils.fairness_metrics import compute_fairness_metrics
from utils.mitigation import mitigate_with_exponentiated_gradient, mitigate_user_model

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

        if not target or not sensitive:
            return jsonify({"error": "target and sensitive are required"}), 400

        df = pd.read_csv(file)
        res = compute_fairness_metrics(df, target, sensitive, pred_col=pred_col)
        return jsonify(res)
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
