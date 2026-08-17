"""
Loads the trained model, vectorizer and scaler once at import time, and
exposes compare_options() - the single function backend/main.py calls for
both /compare and /generate-and-compare, so the two endpoints can never
apply different feature logic.
"""

from pathlib import Path

import joblib


from backend.model_utils import SoftVotingOnlineEnsemble  
from backend.feature_engineering import make_features

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

_model = joblib.load(MODELS_DIR / "social_skills_model.pkl")
_vectorizer = joblib.load(MODELS_DIR / "social_skills_vectorizer.pkl")
_scaler = joblib.load(MODELS_DIR / "social_skills_scaler.pkl")

print(f"[inference] Loaded model: {type(_model).__name__}")
print(f"[inference] Loaded vectorizer: {type(_vectorizer).__name__} "
      f"({len(_vectorizer.get_feature_names_out())} terms)")
print(f"[inference] Loaded scaler: {type(_scaler).__name__}")


def compare_options(prompt: str, option_a: str, option_b: str) -> dict:
    """
    Run the real trained model on (prompt, option_a, option_b) and return
    winner/prediction/probabilities. This is the ONLY place inference-time
    features are built, using the same make_features() function training
    used to fit the vectorizer/scaler - so train/inference cannot drift.
    """
    X = make_features(
        _vectorizer,
        _scaler,
        prompt=prompt,
        response_a=option_a,
        response_b=option_b,
        category="General",
        situation="",
    )

    prediction = int(_model.predict(X)[0])
    probabilities = _model.predict_proba(X)[0]

    return {
        "winner": "Response A" if prediction == 0 else "Response B",
        "prediction": prediction,
        "response_a_probability": round(float(probabilities[0]) * 100, 2),
        "response_b_probability": round(float(probabilities[1]) * 100, 2),
    }
