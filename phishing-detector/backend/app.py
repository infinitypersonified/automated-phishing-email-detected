from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from model_service import ModelService
import logging
import os

app = Flask(__name__)

# 🛡️ THE FIX: Broad CORS policy plus explicit manual handling for tricky browsers
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
}})

# 🚀 CLOUD FETCH: Points to your Hugging Face Repository
HF_REPO_ID = "Infinitypersonified/Automatedphishing-model"

# Initialize the service
try:
    service = ModelService(HF_REPO_ID)
except Exception as e:
    logging.error(f"Failed to load ModelService: {e}")
    service = None

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": service is not None})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # 🕵️ Handle the 'Pre-flight' OPTIONS request sent by the browser
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    if service is None:
        return _corsify_actual_response(jsonify({"error": "Model service loading..."}), 503)

    payload = request.get_json(silent=True) or {}
    email_text = payload.get("email_text", "").strip()

    if not email_text:
        return _corsify_actual_response(jsonify({"error": "email_text is required"}), 400)

    try:
        result = service.predict(email_text)
        return _corsify_actual_response(jsonify(result))
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return _corsify_actual_response(jsonify({"error": str(e)}), 500)

# --- CORS HELPERS ---
def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response

def _corsify_actual_response(response, status=200):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response, status

if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    # debug=False is safer for production to prevent memory spikes
    app.run(debug=False, host="0.0.0.0", port=port)