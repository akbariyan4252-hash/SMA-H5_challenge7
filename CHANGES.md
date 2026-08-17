# What changed in this pass

## Stabilization pass (this update)

Only `backend/gemini_service.py` was touched, in two small, targeted
ways — everything else (all endpoints, the ML pipeline, the frontend,
notebooks, legacy files, tests) is unchanged:

1. **Single `GEMINI_MODEL` constant.** The model name was previously
   hard-coded inline as `"gemini-2.5-flash"` in the `generate_content()`
   call. It's now a single module-level constant,
   `GEMINI_MODEL = "gemini-3.6-flash"`, used everywhere the model name
   is needed — no more risk of two different model strings drifting
   apart.
2. **Server-side timeout on the Gemini call.** `genai.Client(...)` now
   passes `http_options={"timeout": GEMINI_TIMEOUT_MS}` (15s). Previously
   a hung network connection to Gemini had no server-side time limit at
   all, so `/generate` and `/generate-and-compare` could in theory hang
   indefinitely; the client already had a 20s `AbortController` timeout,
   but that only protects the browser tab, not the backend request. The
   backend now times out first (15s) and returns a clean `503` before the
   frontend's own timeout would fire.

Everything else about Gemini's integration is unchanged: it's still
only used by `/generate` / `/generate-and-compare` to produce two
candidate options, the trained ML model still makes every final
comparison, and `/compare` still never imports or touches
`gemini_service` at all.

## Earlier pass

Only the frontend was touched. The backend (`backend/`) and models
(`models/`) are exactly as they were in the uploaded project — they
already worked correctly against the real `.pkl` files (verified by
calling `compare_options(...)` directly).

## `frontend/index.html`

This is now your **original UI** (`index_7-1.html`) with the Social
Skills AI flows wired into the existing Decision Queue. Nothing was
redesigned or replaced — dashboard, history, calibration, settings,
admin, theming, fonts, accessibility mode, voice chat, and the
swipe/approve/adjust/decline mechanics are all untouched.

Concretely:

1. **New category** `social` added next to `inventory` / `risk` /
   `ops` / `support` (color, tag class, FA/EN label) — it plugs into
   the existing category filter/settings toggles automatically.

2. **New collapsible panel** at the top of the Decision Queue view
   ("Start a new decision"), with:
   - a situation textarea + mic button (reuses the app's existing
     `getSpeechRecognizer()` voice helper)
   - a two-way mode switch: **"I need suggestions"** vs. **"I
     already have two options"**
   - Mode 1 → `POST /generate` → shows Option A/B → **"Let AI
     choose"** → `POST /compare`
   - Mode 2 → Option A/B fields → **"Decide for me"** → `POST
     /compare` directly (no Gemini dependency)

3. **`API_BASE_URL = "http://127.0.0.1:8002"`** is the single place
   the base URL is defined; all calls go through one `apiPost()`
   helper with a 20s timeout (`AbortController`), JSON error-body
   parsing, button loading text, and disabled state while a request
   is in flight. Errors surface as toasts (using the app's existing
   `showToast()`) and as inline text in the panel.

4. **`addSocialDecisionToQueue()`** turns a real `/compare` response
   into a card matching the app's existing item schema (title,
   reason, meta, factors, why-this-recommendation, compare strip)
   and unshifts it into the existing `queueItems` array — so it goes
   through the pre-existing Approve / Adjust / Decline / swipe /
   session-log code unmodified. The comparison strip is repurposed
   to show the recommended response vs. the alternative response
   (labelled "Not chosen") instead of a past similar case.

5. **Full FA + EN translations** added for all new panel text,
   following the app's existing `T.fa` / `T.en` pattern.

6. **One small existing-code fix**: the shared card renderer
   computed a bar width as `factor.value * 2`, which assumed values
   under 50 (true for the app's mock data). Real model probabilities
   can exceed that, so it's now clamped to 100%. This doesn't affect
   any existing mock cards.

## Not changed

- `backend/main.py`, `schemas.py`, `gemini_service.py`,
  `inference.py`, `feature_engineering.py`, `model_utils.py` — all
  identical to the uploaded project. `/compare` never depends on
  Gemini; `/generate` and `/generate-and-compare` return a clean
  `503` if `GEMINI_API_KEY` isn't set.
- `models/*.pkl` — untouched, still the real trained model/
  vectorizer/scaler.

## How to run

```bash
cd Social-Skills-AI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: add GEMINI_API_KEY to enable "I need suggestions"

python -m uvicorn backend.main:app --reload --port 8002
```

In a second terminal:

```bash
python -m http.server 5500 --directory frontend
```

Open **http://127.0.0.1:5500**, go to the **Decision Queue**, and
use the new panel at the top.

## Testing both flows

**Flow 1 — no options yet:**
Type "I want to start a conversation with a new person." → click
**"I need suggestions"** → **Generate suggestions** → review Option
A/B → **Let AI choose**. A new card appears at the top of the
Decision Queue with the real model's winner and confidence.

**Flow 2 — already have two options:**
Click **"I already have two options"** → fill in Situation, Option
A, Option B → **Decide for me**. Same result, no Gemini call made.

**Flow 3 — Gemini unavailable:**
Remove/omit `GEMINI_API_KEY` from `.env`. Flow 1's "Generate
suggestions" now shows a clear error toast (from the backend's
`503`). Flow 2 keeps working normally, since `/compare` never
touches Gemini.
