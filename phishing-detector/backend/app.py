from flask import Flask, jsonify, request
from flask_cors import CORS
from model_service import ModelService
import logging

app = Flask(__name__)

# 🛡️ FIXED: Wildcard CORS to allow your Render frontend to talk to this backend
CORS(app, resources={r"/*": {"origins": "*"}})

# 🚀 CLOUD FETCH: Points to your Hugging Face Repository
HF_REPO_ID = "Infinitypersonified/Automatedphishing-model"

# Initialize the service
# Note: This might take a moment to load on Render's first start
try:
    service = ModelService(HF_REPO_ID)
except Exception as e:
    logging.error(f"Failed to load ModelService: {e}")
    service = None

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": service is not None})

@app.route("/predict", methods=["POST"])
def predict():
    if service is None:
        return jsonify({"error": "Model service is not initialized. Please wait or check logs."}), 503

    payload = request.get_json(silent=True) or {}
    email_text = payload.get("email_text", "").strip()

    if not email_text:
        return jsonify({"error": "email_text is required"}), 400

    try:
        # This calls the service.predict we fixed in the last step
        result = service.predict(email_text)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render uses the PORT environment variable, so we use 10000 as a default
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)