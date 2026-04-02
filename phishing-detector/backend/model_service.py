from __future__ import annotations
import torch
import gc  # Garbage Collector
from typing import Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from feature_extraction import compute_structural_features

class ModelService:
    def __init__(self, hf_repo_id: str):
        self.hf_repo_id = hf_repo_id
        
        # 1. Load Tokenizer (Lightweight)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_repo_id)
        
        # 2. 🛡️ MEMORY FIX: Load model with low_cpu_mem_usage
        # We use 'cpu' explicitly because Render Free Tier doesn't have a GPU (CUDA)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hf_repo_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32 # Use float32 on CPU for stability
        )

        self.device = torch.device("cpu") 
        self.model.to(self.device)
        self.model.eval()
        
        # 3. 🧹 MEMORY FIX: Clear RAM immediately after loading
        gc.collect()

    def predict(self, text: str) -> dict[str, Any]:
        text_lower = text.lower()

        # 🔥 RULE 1: Legit overrides (Saves RAM by skipping the model)
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

        # 🚨 RULE 2: Phishing overrides (Saves RAM by skipping the model)
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

            # ✅ GROUP 1: TRUSTED SYSTEM & DEV OPS (Prevents your Render/GitHub trap)
        if any(keyword in text_lower for keyword in [
            "deploy process", "build successful", "commit:", "pull request", 
            "github", "render.com", "vercel", "deployment live", 
            "npm install", "dependency", "merged into main"
        ]):
            return {
                "prediction": "legitimate",
                "probability": 0.05,
                "explanation": "Verified as a technical system/deployment notification."
            }

        # ✅ GROUP 2: TRANSACTIONAL & BANKING (Common safe "Urgent" emails)
        if any(keyword in text_lower for keyword in [
            "one-time password", "otp is:", "your verification code", 
            "transaction successful", "debited from your account", 
            "receipt for your purchase", "order #", "tracking number"
        ]):
            return {
                "prediction": "legitimate",
                "probability": 0.1,
                "explanation": "Recognized as a standard transactional or security code message."
            }

        # 🚨 GROUP 3: INSTANT RED FLAGS (Hard Phishing - No AI needed)
        # These are so obvious we don't want to risk the AI saying "Maybe safe"
        if any(keyword in text_lower for keyword in [
            "kindly provide your ssn", "send your password to", 
            "account will be deleted in 1 hour", "win a $1000 gift card",
            "inherited millions", "western union transfer needed",
            "verify your identity here:", "login-update-required.php"
        ]):
            return {
                "prediction": "phishing",
                "probability": 1.0,
                "explanation": "Matched high-risk phishing patterns (Urgent threats or credential harvesting)."
            }

        # 🤖 FALLBACK → MODEL
        features = compute_structural_features(text)

        # 4. 🛡️ MEMORY FIX: Reduce max_length to 128 to save memory during processing
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128, # Shorter length = less RAM used
            return_token_type_ids=False
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs.pop("token_type_ids", None)

        # 5. 🛡️ MEMORY FIX: Use 'no_grad' to prevent torch from building a memory-heavy "graph"
        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        phishing_prob = probs[0][1].item()

        prediction = "phishing" if phishing_prob >= 0.8 else "legitimate"

        explanation_bits = []
        if features.get("suspicious_words"):
            explanation_bits.append("suspicious keyword(s) detected")
        if features.get("num_links", 0) > 0:
            explanation_bits.append("contains links")
        
        explanation = (
            "⚠️ " + ", ".join(explanation_bits)
            if explanation_bits
            else "No strong indicators detected, but intention suggests phishing"
        )

        # 6. 🧹 Final RAM cleanup
        del inputs
        del outputs
        gc.collect()

        return {
            "prediction": prediction,
            "probability": round(phishing_prob, 4),
            "suspicious_words": features.get("suspicious_words", []),
            "links_detected": features.get("links_detected", []),
            "explanation": explanation
        }