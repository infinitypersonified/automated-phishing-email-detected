from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from model_service import ModelService
import logging
import os

app = Flask(__name__)

# 🛡️ THE FIX: Simplified but powerful CORS configuration
# Using 'supports_credentials' and explicit origins helps browsers trust the link
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
}})

# 🚀 CLOUD FETCH: Points to your Hugging Face Repository
HF_REPO_ID = "Infinitypersonified/Automatedphishing-model"

# Initialize the service globally
service = None
try:
    service = ModelService(HF_REPO_ID)
except Exception as e:
    logging.error(f"CRITICAL: Failed to load ModelService: {e}")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok", 
        "model_loaded": service is not None
    }), 200

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # 🕵️ Manual Handle for 'Pre-flight' (though CORS(app) usually handles this)
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    if service is None:
        return _corsify_response(jsonify({"error": "Model not loaded on server"}), 503)

    payload = request.get_json(silent=True) or {}
    email_text = payload.get("email_text", "").strip()

    if not email_text:
        return _corsify_response(jsonify({"error": "email_text is required"}), 400)

    try:
        result = service.predict(email_text)
        return _corsify_response(jsonify(result), 200)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return _corsify_response(jsonify({"error": str(e)}), 500)

# --- REFINED CORS HELPERS ---
def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
    return response

def _corsify_response(response_obj, status_code):
    # This ensures the header is attached to the actual response object before the status code
    response_obj.headers.add("Access-Control-Allow-Origin", "*")
    return response_obj, status_code

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)