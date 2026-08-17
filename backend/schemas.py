"""Pydantic request/response schemas for the Social Skills AI API."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    message: str
    gemini_available: bool


class CompareRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The social situation.")
    option_a: str = Field(..., min_length=1, description="Response / option A.")
    option_b: str = Field(..., min_length=1, description="Response / option B.")


class CompareResponse(BaseModel):
    prompt: str
    option_a: str
    option_b: str
    winner: str
    prediction: int
    response_a_probability: float
    response_b_probability: float


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The social situation.")


class GenerateResponse(BaseModel):
    prompt: str
    option_a: str
    option_b: str


class GenerateAndCompareResponse(BaseModel):
    prompt: str
    option_a: str
    option_b: str
    winner: str
    prediction: int
    response_a_probability: float
    response_b_probability: float
