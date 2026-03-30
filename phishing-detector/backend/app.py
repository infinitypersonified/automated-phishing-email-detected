from flask import Flask, jsonify, request
from flask_cors import CORS
from model_service import ModelService

app = Flask(__name__)
CORS(app)

# 🚀 CLOUD FETCH: Points to your Hugging Face Repository
# This replaces the old MODEL_DIR logic
HF_REPO_ID = "Infinitypersonified/Automatedphishing-model/fine_tuned_phishing_model"

# Initialize the service using the Hugging Face ID
service = ModelService(HF_REPO_ID)

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    email_text = payload.get("email_text", "").strip()

    if not email_text:
        return jsonify({"error": "email_text is required"}), 400

    result = service.predict(email_text)
    return jsonify(result)

if __name__ == "__main__":
    # Note: debug=True will reload the model every time you save a file.
    # On slower internet, you might want to turn it off during testing.
    app.run(debug=True, host="0.0.0.0", port=5000)