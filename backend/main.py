"""
Social Skills AI - FastAPI backend.

Routes only. Model loading lives in backend/inference.py, feature
engineering in backend/feature_engineering.py, the model class in
backend/model_utils.py, and the optional Gemini integration in
backend/gemini_service.py.

Run from the project root with:
    python -m uvicorn backend.main:app --reload --port 8002
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend import gemini_service
from backend.gemini_service import GeminiUnavailableError
from backend.inference import compare_options
from backend.schemas import (
    CompareRequest,
    CompareResponse,
    GenerateAndCompareResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)

app = FastAPI(
    title="Social Skills AI API",
    description="AI system for social-skill decision support",
    version="1.0.0",)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)




@app.get("/", response_model=HealthResponse)
def home():
    return HealthResponse(
        status="online",
        message="Social Skills AI API is running",
        gemini_available=gemini_service.is_available(),
    )




@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    result = compare_options(req.prompt, req.option_a, req.option_b)
    return CompareResponse(
        prompt=req.prompt,
        option_a=req.option_a,
        option_b=req.option_b,
        **result,
    )




@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        option_a, option_b = gemini_service.generate_responses(req.prompt)
    except GeminiUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return GenerateResponse(prompt=req.prompt, option_a=option_a, option_b=option_b)


@app.post("/generate-and-compare", response_model=GenerateAndCompareResponse)
def generate_and_compare(req: GenerateRequest):
    try:
        option_a, option_b = gemini_service.generate_responses(req.prompt)
    except GeminiUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    result = compare_options(req.prompt, option_a, option_b)
    return GenerateAndCompareResponse(
        prompt=req.prompt,
        option_a=option_a,
        option_b=option_b,
        **result,
    )
