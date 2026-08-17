"""
Optional Gemini integration used only by /generate and /generate-and-compare.

The core /compare endpoint (comparing two user-provided options with the
trained ML model) NEVER imports or depends on anything in this module.

If GEMINI_API_KEY is not set, or the google-genai package isn't installed,
or the API call fails for any reason, `generate_responses` raises
GeminiUnavailableError with a human-readable message. The API layer turns
that into a clean HTTP error - it never fabricates fake "offline" options.
"""

import os
import re
from google.genai import types
from dotenv import load_dotenv

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


GEMINI_MODEL = "gemini-3.6-flash"


GEMINI_TIMEOUT_MS = 15000

SYSTEM_PROMPT = """You are a social skills assistant.

The user will describe a social situation.

Generate two different but reasonable responses.

Response A should be practical and direct.
Response B should be empathetic and detailed.

Both responses must be helpful.
Do not say which response is better.

Return the result in this exact format:

RESPONSE_A:
...

RESPONSE_B:
..."""


class GeminiUnavailableError(Exception):
    """Raised whenever the AI-option-generation feature cannot be used."""


def _get_client():
    if not GEMINI_API_KEY:
        raise GeminiUnavailableError(
            "GEMINI_API_KEY is not set. Set it in your .env file to enable "
            "AI-generated options. The /compare endpoint does not require it."
        )
    try:
        from google import genai
    except ImportError as exc:
        raise GeminiUnavailableError(
            "The 'google-genai' package is not installed. "
            "Run: pip install google-genai"
        ) from exc

    try:
        return genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=60000))
    
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean API error
        raise GeminiUnavailableError(f"Could not create Gemini client: {exc}") from exc


def _parse_gemini_response(raw_text: str):
    match = re.search(
        r"RESPONSE_A:\s*(.*?)\s*RESPONSE_B:\s*(.*)",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        raise GeminiUnavailableError(
            "Could not parse RESPONSE_A / RESPONSE_B from the Gemini output."
        )
    return match.group(1).strip(), match.group(2).strip()


def is_available() -> bool:
    """Cheap check for whether generation is configured (used by GET /)."""
    return bool(GEMINI_API_KEY)


def generate_responses(prompt: str) -> tuple[str, str]:
    """
    Generate (response_a, response_b) for a social situation using Gemini.

    Raises GeminiUnavailableError if the key is missing, the client can't
    be created, the API call fails, or the response can't be parsed.
    """
    client = _get_client()

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"system_instruction": SYSTEM_PROMPT},
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean API error
        raise GeminiUnavailableError(f"Gemini API call failed: {exc}") from exc

    return _parse_gemini_response(response.text)
