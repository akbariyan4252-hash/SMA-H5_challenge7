"""
API-level tests using FastAPI's TestClient.

Requires the project's requirements to be installed:
    pip install -r requirements.txt

Run with:
    python -m pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from backend.main import app  # noqa: E402

client = TestClient(app)


def test_health_check():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "gemini_available" in data


def test_compare_valid_input():
    res = client.post("/compare", json={
        "prompt": "I want to start a conversation with a new person.",
        "option_a": "Ask about their day and then introduce myself.",
        "option_b": "Ask how they are and then introduce myself.",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["winner"] in ("Response A", "Response B")
    assert data["prediction"] in (0, 1)


def test_compare_probabilities_valid_and_sum_to_100():
    res = client.post("/compare", json={
        "prompt": "My coworker keeps interrupting me in meetings.",
        "option_a": "Politely ask them to let me finish my point.",
        "option_b": "Say nothing and hope it stops on its own.",
    })
    data = res.json()
    a, b = data["response_a_probability"], data["response_b_probability"]
    assert 0 <= a <= 100
    assert 0 <= b <= 100
    assert abs((a + b) - 100) < 0.5


def test_compare_rejects_missing_fields():
    res = client.post("/compare", json={"prompt": "Missing options"})
    assert res.status_code == 422


def test_generate_fails_gracefully_without_gemini_key(monkeypatch):
    monkeypatch.setattr("backend.gemini_service.GEMINI_API_KEY", None)
    res = client.post("/generate", json={"prompt": "I have a job interview tomorrow."})
    assert res.status_code == 503
    assert "detail" in res.json()


def test_compare_still_works_when_gemini_unavailable(monkeypatch):
    monkeypatch.setattr("backend.gemini_service.GEMINI_API_KEY", None)
    res = client.post("/compare", json={
        "prompt": "I have a job interview tomorrow.",
        "option_a": "Prepare answers to common questions in advance.",
        "option_b": "Wing it and hope for the best.",
    })
    assert res.status_code == 200


def test_frontend_can_call_backend_cors_headers_present():
    res = client.options(
        "/compare",
        headers={
            "Origin": "http://127.0.0.1:5500",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert res.status_code in (200, 204)
    assert res.headers.get("access-control-allow-origin") in ("*", "http://127.0.0.1:5500")
