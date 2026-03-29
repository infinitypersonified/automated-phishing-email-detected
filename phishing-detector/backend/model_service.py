from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from feature_extraction import compute_structural_features

class ModelService:
    def __init__(self, hf_repo_id: str):
        # 🤖 Instead of looking for a folder on your C: drive, 
        # we now look for your repo ID on Hugging Face.
        self.hf_repo_id = hf_repo_id

        # This will automatically download the files from the cloud
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_repo_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.hf_repo_id)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict[str, Any]:
        text_lower = text.lower()

        # 🔥 RULE 1: Legit overrides
        if any(keyword in text_lower for keyword in [
            "otp", "do not share", "transaction", "transfer", 
            "current balance", "dear customer", "thank you"
        ]):
            return {
                "prediction": "legitimate",
                "probability": 0.0,
                "suspicious_words": [],
                "links_detected": [],
                "explanation": "Recognized as a normal system-generated message"
            }

        # 🚨 RULE 2: Phishing overrides
        if any(keyword in text_lower for keyword in [
            "click here", "urgent", "verify your account", 
            "you have won", "suspended", "claim your reward",
        ]):
            return {
                "prediction": "phishing",
                "probability": 1.0,
                "suspicious_words": ["trigger phrase"],
                "links_detected": [],
                "explanation": "Matched common phishing pattern"
            }

        # 🤖 FALLBACK → MODEL
        features = compute_structural_features(text)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        phishing_prob = probs[0][1].item()

        prediction = "phishing" if phishing_prob >= 0.9 else "legitimate"

        explanation_bits = []
        if features["suspicious_words"]:
            explanation_bits.append("suspicious keyword(s) detected")
        if features["num_links"] > 0:
            explanation_bits.append("contains links")
        if features["sender_mismatch"]:
            explanation_bits.append("sender mismatch detected")
        if features["has_ip_url"]:
            explanation_bits.append("IP-based URL detected")

        explanation = (
            "⚠️ " + ", ".join(explanation_bits)
            if explanation_bits
            else "No strong phishing indicators detected but phishing intention detected"
        )

        return {
            "prediction": prediction,
            "probability": round(phishing_prob, 4),
            "suspicious_words": features["suspicious_words"],
            "links_detected": features["links_detected"],
            "explanation": explanation
        }