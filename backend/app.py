from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from utils.fairness_metrics import compute_fairness_metrics

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        target = request.form.get("target")
        sensitive = request.form.get("sensitive")

        if not target or not sensitive:
            return jsonify({"error": "Missing target or sensitive attribute"}), 400

        df = pd.read_csv(file)
        results = compute_fairness_metrics(df, target, sensitive)
        return jsonify(results)

    except Exception as e:
        print("❌ Backend error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
