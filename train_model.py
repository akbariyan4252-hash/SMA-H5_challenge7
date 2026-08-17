"""
Trains the Social Skills AI comparison model and saves artifacts to models/.

This reproduces the training pipeline from the original notebook exactly
(same dataset, same train/val/test split with random_state=42, same
TfidfVectorizer settings, same MaxAbsScaler, same SoftVotingOnlineEnsemble
of SGDClassifier(log_loss) + SGDClassifier(modified_huber) + MultinomialNB).

The only fix vs. the original notebook: the model class is imported from
backend.model_utils (an importable module) instead of being redefined in
__main__ / a notebook cell, so the saved .pkl is portable and loads cleanly
under `uvicorn backend.main:app`.

Run from the project root:
    python train_model.py

This is also the script notebooks/training.ipynb mirrors cell-by-cell.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MaxAbsScaler

from backend.feature_engineering import build_combined_text, build_numeric_features
from backend.model_utils import SoftVotingOnlineEnsemble

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "social_skills_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main():
    # ---- 1. Load dataset ----
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # ---- 2. Validate ----
    required_cols = {"category", "situation", "prompt", "response_a", "response_b", "human_choice"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    null_counts = df.isnull().sum()
    if null_counts.any():
        raise ValueError(f"Dataset contains nulls:\n{null_counts[null_counts > 0]}")

    print("\nCategory distribution:")
    print(df["category"].value_counts())
    print("\nHuman choice distribution:")
    print(df["human_choice"].value_counts(normalize=True))

    # ---- 3. Feature engineering (shared module - identical to inference) ----
    df["combined_text"] = build_combined_text(
        df["category"], df["situation"], df["prompt"], df["response_a"], df["response_b"]
    )
    numeric_features = build_numeric_features(df["response_a"], df["response_b"])

    x_text = df["combined_text"]
    y = df["human_choice"]

    # ---- 4. Train/val/test split (stratified, same random_state as original) ----
    idx_trainval, idx_test = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=y
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.25, random_state=42, stratify=y.loc[idx_trainval]
    )
    print(f"\nTrain: {len(idx_train)} | Validation: {len(idx_val)} | Test: {len(idx_test)}")

    # ---- 5. TF-IDF: fit on train only, transform val/test ----
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)
    x_train_tfidf = vectorizer.fit_transform(x_text.loc[idx_train])
    x_val_tfidf = vectorizer.transform(x_text.loc[idx_val])
    x_test_tfidf = vectorizer.transform(x_text.loc[idx_test])

    # ---- 6. Numeric features: scale, fit on train only ----
    numeric_scaler = MaxAbsScaler()
    num_train = numeric_scaler.fit_transform(numeric_features.loc[idx_train])
    num_val = numeric_scaler.transform(numeric_features.loc[idx_val])
    num_test = numeric_scaler.transform(numeric_features.loc[idx_test])

    x_train_full = sp.hstack([x_train_tfidf, num_train]).tocsr()
    x_val_full = sp.hstack([x_val_tfidf, num_val]).tocsr()
    x_test_full = sp.hstack([x_test_tfidf, num_test]).tocsr()

    y_train, y_val, y_test = y.loc[idx_train], y.loc[idx_val], y.loc[idx_test]

    # ---- 7. Train the ensemble ----
    model = SoftVotingOnlineEnsemble([
        ("sgd_log", SGDClassifier(loss="log_loss", random_state=42)),
        ("sgd_huber", SGDClassifier(loss="modified_huber", random_state=7)),
        ("naive_bayes", MultinomialNB()),
    ])
    model.fit(x_train_full, y_train)

    # ---- 8. Evaluate ----
    val_pred = model.predict(x_val_full)
    print("\nValidation accuracy:", f"{accuracy_score(y_val, val_pred):.2%}")

    y_pred = model.predict(x_test_full)
    print("Test accuracy:", f"{accuracy_score(y_test, y_pred):.2%}")
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, target_names=["Response A", "Response B"]))
    print("Confusion matrix (test set):")
    print(confusion_matrix(y_test, y_pred))

    # ---- 9. Sanity-check test prediction using the real inference path ----
    from backend.feature_engineering import make_features

    sample = make_features(
        vectorizer, numeric_scaler,
        prompt="I want to start a conversation with a new person.",
        response_a="Ask about their day and then introduce myself.",
        response_b="Ask how they are and then introduce myself.",
    )
    sample_pred = model.predict(sample)[0]
    sample_proba = model.predict_proba(sample)[0]
    print("\nSanity check prediction:", sample_pred, "probs:", sample_proba)

    # ---- 10. Save artifacts ----
    joblib.dump(model, MODELS_DIR / "social_skills_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "social_skills_vectorizer.pkl")
    joblib.dump(numeric_scaler, MODELS_DIR / "social_skills_scaler.pkl")
    print(f"\nSaved model, vectorizer, and scaler to {MODELS_DIR}")


if __name__ == "__main__":
    main()
