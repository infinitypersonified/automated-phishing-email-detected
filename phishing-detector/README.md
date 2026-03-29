# Phishing Detector (Full-Stack)

A research-focused phishing email detection system with:

- Flask backend (`/predict` endpoint)
- React + Vite frontend dashboard
- Traditional ML models (Logistic Regression, Random Forest, SVM)
- Transformer model option (DistilBERT with HuggingFace)
- Explainability signals (suspicious keywords, links, sender mismatch, URL/IP indicators)

## Project Structure

```text
phishing-detector/
  frontend/
  backend/
  model/
  dataset/
  training/
  README.md
```

## 1) Backend Setup and Run

From `phishing-detector/backend`:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell
pip install -r requirements.txt
python app.py
```

Backend default URL: `http://127.0.0.1:5000`

## 2) Frontend Setup and Run

From `phishing-detector/frontend`:

```bash
npm install
npm run dev
```

Frontend default URL: `http://127.0.0.1:5173`

## 3) Model Training

Place dataset in `phishing-detector/dataset/emails.csv` with columns:

- `text`: email body/content
- `label`: `phishing` or `legitimate` (also accepts `1/0`)

Run traditional model training (recommended first):

```bash
cd phishing-detector/training
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell
pip install -r requirements.txt
python train_models.py --dataset ../dataset/emails.csv --output ../model
```

Optional DistilBERT fine-tuning:

```bash
python train_distilbert.py --dataset ../dataset/emails.csv --output ../model/distilbert_model
```

## 4) API Contract

`POST /predict`

Request:

```json
{
  "email_text": "Your email text here..."
}
```

Response:

```json
{
  "prediction": "phishing",
  "probability": 0.92,
  "suspicious_words": ["urgent", "verify"],
  "links_detected": ["http://example.com/verify"],
  "extracted_features": {
    "num_links": 1,
    "num_domains": 1,
    "url_avg_length": 25.0,
    "has_ip_url": 0,
    "sender_mismatch": 1
  },
  "explanation": "The email contains suspicious keywords and multiple risk indicators."
}
```

## 5) Notes for Research Defense Demo

- Train models before demo and keep artifacts in `model/`
- Backend auto-loads best available model bundle (`traditional_model_bundle.joblib`)
- Frontend highlights suspicious words in input content
- Use diverse test samples (invoice fraud, credential reset scams, normal newsletters)

