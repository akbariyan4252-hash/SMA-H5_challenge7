from model_utils import SoftVotingOnlineEnsemble
from pathlib import Path
import joblib

from fastapi import FastAPI
from pydantic import BaseModel


# ==========================================
# 1. Create FastAPI application
# ==========================================

app = FastAPI(
    title="Social Skills AI API",
    description="AI system for social-skill decision support",
    version="1.0.0"
)


# ==========================================
# Load ML components
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

model = joblib.load(MODELS_DIR / "social_skills_model.pkl")
vectorizer = joblib.load(MODELS_DIR / "social_skills_vectorizer.pkl")
scaler = joblib.load(MODELS_DIR / "social_skills_scaler.pkl")

print("Model loaded:", type(model))
print("Vectorizer loaded:", type(vectorizer))
print("Scaler loaded:", type(scaler))


# ==========================================
# 3. Request schemas
# ==========================================

class UserInput(BaseModel):
    text: str


class CompareRequest(BaseModel):
    prompt: str
    option_a: str
    option_b: str


# ==========================================
# 4. Health check
# ==========================================

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Social Skills AI API is running"
    }


# ==========================================
# 5. Single prediction
# ==========================================

@app.post("/predict")
def predict(data: UserInput):

    X = vectorizer.transform([data.text])

    prediction = model.predict(X)

    result = int(prediction[0])

    response = {
        "input": data.text,
        "prediction": result
    }

    # Probability if model supports it
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]

        response["response_a_probability"] = round(
            float(probabilities[0]) * 100, 2
        )

        response["response_b_probability"] = round(
            float(probabilities[1]) * 100, 2
        )

        response["label"] = (
            "Response A" if result == 0 else "Response B"
        )

    return response


# ==========================================
# 6. Compare two user-provided options
# ==========================================

@app.post("/compare")
def compare(req: CompareRequest):

    text_a = req.prompt + " " + req.option_a
    text_b = req.prompt + " " + req.option_b

    X = vectorizer.transform([
        text_a,
        text_b
    ])

    probabilities = model.predict_proba(X)

    prob_a = float(probabilities[0][1])
    prob_b = float(probabilities[1][1])

    if prob_a >= prob_b:
        winner = "Option A"
    else:
        winner = "Option B"

    return {
        "prompt": req.prompt,
        "option_a": req.option_a,
        "option_b": req.option_b,
        "winner": winner,
        "option_a_probability": round(prob_a * 100, 2),
        "option_b_probability": round(prob_b * 100, 2)
    }