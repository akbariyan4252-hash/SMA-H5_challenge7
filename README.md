# Social Skills AI

An AI system for social-skill decision support. Given a **situation** and two
candidate **responses**, a trained ML model predicts which response people
tend to prefer — and by how much. Optionally, an LLM (Gemini) can generate
the two candidate responses for you first.

```
USER SITUATION
      |
      v
(optional) LLM generates Option A + Option B  --- /generate
      |
      v
ML comparison model                            --- /compare
      |
      v
Winner + probabilities
      |
      v
Frontend
```

`/compare` (the core feature) never depends on Gemini. If `GEMINI_API_KEY`
isn't set, `/generate` and `/generate-and-compare` return a clean HTTP error;
everything else keeps working.

---

## Architecture

```
Social-Skills-AI/
│
├── backend/
│   ├── __init__.py
│   ├── main.py                 FastAPI routes only
│   ├── schemas.py               Pydantic request/response models
│   ├── model_utils.py           SoftVotingOnlineEnsemble (importable, not __main__)
│   ├── feature_engineering.py   Shared train+inference feature logic
│   ├── inference.py             Loads model/vectorizer/scaler, compare_options()
│   └── gemini_service.py        Optional Gemini integration
│
├── frontend/
│   ├── index.html                Two-mode UI (compare / AI-suggest)
│   ├── style.css
│   └── script.js                 Talks to the backend via fetch()
│
├── models/
│   ├── social_skills_model.pkl
│   ├── social_skills_vectorizer.pkl
│   └── social_skills_scaler.pkl
│
├── notebooks/
│   └── training.ipynb            Training only — never runs the API
│
├── data/
│   └── social_skills_dataset.csv
│
├── tests/
│   ├── test_inference.py         No FastAPI required
│   └── test_api.py               Requires fastapi + httpx
│
├── legacy/                       Original files, archived with notes (see legacy/README.md)
│
├── train_model.py                Script version of the training notebook
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ML approach

**Task**: binary classification. Given `(situation, response_a, response_b)`,
predict which response a human would prefer (`0` = A, `1` = B).

**Dataset**: `data/social_skills_dataset.csv` — 2,016 rows, columns
`category, situation, prompt, response_a, response_b, human_choice`, 12
social-skill categories (Public Speaking, Handling Criticism, Assertiveness,
etc.), no nulls, no duplicates, roughly balanced classes (~52% / ~48%).

**Feature engineering** (`backend/feature_engineering.py` — the single shared
implementation used by both training and inference):

- **Text**: `combined_text = "CATEGORY: {category} SITUATION: {situation} PROMPT: {prompt} RESPONSE_A: {response_a} RESPONSE_B: {response_b}"`,
  vectorized with `TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)`.
- **Numeric** (10 features, scaled with `MaxAbsScaler`): `char_len_a`,
  `word_count_a`, `has_question_a`, `has_exclaim_a`, `char_len_b`,
  `word_count_b`, `has_question_b`, `has_exclaim_b`, `length_diff_abs`,
  `a_is_longer`. **Response A and Response B statistics are always computed
  separately, from their own text** — never from a shared/combined string.
  This matters: an earlier inference helper in the original notebook
  computed stats once and duplicated them into both slots, which silently
  broke the numeric features at prediction time.
- Final feature matrix: `hstack([tfidf, scaled_numeric])`.

**Model**: `SoftVotingOnlineEnsemble` (`backend/model_utils.py`) — soft-votes
across three online-capable ("partial_fit"-able) classifiers:
`SGDClassifier(loss="log_loss")`, `SGDClassifier(loss="modified_huber")`, and
`MultinomialNB()`.

**Split**: stratified, `random_state=42` — 60% train / 20% validation / 20%
test. Vectorizer and scaler are `fit` on train only.

**Current performance** (see `notebooks/training.ipynb` for the full run):
validation accuracy **87.3%**, test accuracy **91.3%**
(precision/recall/f1 ≈ 0.91 for both classes).

**Why the model was retrained rather than reused as-is**: the `.pkl` files
you uploaded were trained with `SoftVotingOnlineEnsemble` defined inline in a
notebook (`__main__`), so `joblib.load(...)` outside Jupyter fails with
`AttributeError: Can't get attribute 'SoftVotingOnlineEnsemble' on <module
'__main__'>`. The model was retrained with the **exact same pipeline,
hyperparameters, and random_state** — only the class's *module location*
changed (now `backend.model_utils`), so the ML logic is unchanged and the
saved model is portable. See `legacy/README.md` for full details.

---

## Installation

```bash
git clone <your-repo-url>
cd Social-Skills-AI
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # optional — only needed for /generate
```

## Training

Model artifacts in `models/` are already trained and committed, so you don't
need to retrain to run the app. To retrain (e.g. after updating the dataset):

```bash
python train_model.py
```

or open `notebooks/training.ipynb` in Jupyter and run all cells. Both do the
exact same thing; the notebook has explanatory markdown, the script is meant
for CI / quick reruns.

## Running the backend

From the project root (not from inside `backend/`):

```bash
python -m uvicorn backend.main:app --reload --port 8002
```

Swagger docs: **http://127.0.0.1:8002/docs**

## Running the frontend

In a second terminal:

```bash
python -m http.server 5500 --directory frontend
```

Open **http://127.0.0.1:5500**

The frontend talks to the backend at `http://127.0.0.1:8002` (set once, in
`frontend/script.js`, as `API_BASE_URL`).

---

## API endpoints

### `GET /`
Health check.

```json
{
  "status": "online",
  "message": "Social Skills AI API is running",
  "gemini_available": false
}
```

### `POST /compare` — core endpoint, no Gemini required

Request:

```json
{
  "prompt": "I want to start a conversation with a new person.",
  "option_a": "Ask about their day and then introduce myself.",
  "option_b": "Ask how they are and then introduce myself."
}
```

Response:

```json
{
  "prompt": "I want to start a conversation with a new person.",
  "option_a": "Ask about their day and then introduce myself.",
  "option_b": "Ask how they are and then introduce myself.",
  "winner": "Response A",
  "prediction": 0,
  "response_a_probability": 76.79,
  "response_b_probability": 23.21
}
```

### `POST /generate` — optional, requires `GEMINI_API_KEY`

Request: `{"prompt": "My coworker keeps interrupting me in meetings."}`

Response: `{"prompt": "...", "option_a": "...", "option_b": "..."}`

If `GEMINI_API_KEY` is missing or the API call fails, returns `503` with a
`detail` message. It does not fabricate fake "offline" options.

### `POST /generate-and-compare` — optional, requires `GEMINI_API_KEY`

Request: `{"prompt": "..."}`

Response: same shape as `/compare`'s response, with the generated
`option_a` / `option_b` included.

---

## Environment variables

| Variable          | Required? | Purpose                                    |
|--------------------|-----------|---------------------------------------------|
| `GEMINI_API_KEY`   | No        | Enables `/generate` and `/generate-and-compare`. `/compare` works without it. |

Never commit a real key — `.env` is git-ignored; use `.env.example` as the
template.

---

## Testing

```bash
# ML core only — no FastAPI required
python tests/test_inference.py

# Full API tests — requires `pip install -r requirements.txt`
python -m pytest tests/test_api.py -v
```

`tests/test_inference.py` covers: model/vectorizer/scaler load successfully,
`/compare`-equivalent predictions return a valid winner and probabilities
that sum to ~100, and — critically — that Response A and Response B numeric
features are computed independently (not duplicated from one string).

`tests/test_api.py` covers: health check, valid `/compare` input,
probability bounds/sum, `422` on missing fields, `/generate` failing
gracefully (`503`) without a Gemini key, and `/compare` continuing to work
when Gemini is unavailable.

---

## Limitations

- The dataset is 2,016 examples across 12 categories — solid for a v1, but
  small enough that performance on situations very different from the
  training categories may be weaker.
- `GEMINI_API_KEY` is optional; without it, only `/compare` (manual options)
  is available — `/generate` and `/generate-and-compare` return `503`.
- The model is a static snapshot from `models/`. There's no online
  learning / feedback loop wired into the production API (the original
  notebook had an experimental `partial_fit`-based feedback endpoint — see
  `legacy/README.md` — which could be reintroduced later with proper
  persistence and concurrency handling).
- CORS is currently open (`allow_origins=["*"]`) for local development
  convenience; restrict this before deploying publicly.
