from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from model_service import ModelService
import logging
import os

app = Flask(__name__)

# 1. Standard CORS setup
CORS(app, resources={r"/*": {"origins": "*"}})

# 🚀 Load the model
HF_REPO_ID = "Infinitypersonified/Automatedphishing-model"
service = None
try:
    service = ModelService(HF_REPO_ID)
except Exception as e:
    logging.error(f"Model Load Failed: {e}")

# 2. THE NUCLEAR OPTION: Force headers on EVERY response
@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    return response

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": service is not None})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Handle the browser's "Pre-flight" check
    if request.method == "OPTIONS":
        return make_response("", 204)

    if service is None:
        return jsonify({"error": "Model not ready"}), 503

    payload = request.get_json(silent=True) or {}
    email_text = payload.get("email_text", "").strip()

    if not email_text:
        return jsonify({"error": "email_text is required"}), 400

    try:
        result = service.predict(email_text)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)