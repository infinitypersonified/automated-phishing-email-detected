from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from feature_builder import build_features_dataframe


def normalize_label(value) -> int:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"phishing", "spam", "1", "true"}:
            return 1
        return 0
    return int(value)


def evaluate_model(name, model, X_test, y_test):
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    print(f"\n=== {name} ===")
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset CSV")
    parser.add_argument("--output", required=True, help="Output model directory")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")

    df["text"] = df["text"].fillna("")
    df["label_num"] = df["label"].apply(normalize_label)

    feature_df = build_features_dataframe(df["text"].tolist())
    feature_cols = [
        "num_links",
        "num_domains",
        "num_suspicious_words",
        "url_avg_length",
        "has_ip_url",
        "sender_mismatch",
    ]

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(df["text"])
    X_struct = csr_matrix(feature_df[feature_cols].values)
    X = hstack([X_tfidf, X_struct])
    y = df["label_num"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
        "svm": SVC(kernel="linear", probability=True, class_weight="balanced"),
    }

    report = {}
    best_name = None
    best_f1 = -1.0
    best_model = None
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(name, model, X_test, y_test)
        report[name] = metrics
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = name
            best_model = model

    bundle = {
        "vectorizer": vectorizer,
        "model": best_model,
        "model_name": best_name,
        "feature_order": feature_cols,
    }
    bundle_path = output_dir / "traditional_model_bundle.joblib"
    joblib.dump(bundle, bundle_path)
    (output_dir / "metrics_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved best model ({best_name}) to: {bundle_path}")


if __name__ == "__main__":
    main()
