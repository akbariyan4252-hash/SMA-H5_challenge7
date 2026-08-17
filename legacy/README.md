# Legacy / archived files

These are the original files you uploaded, kept for reference but **not** part
of the clean production app. Here's what each one is and why it's here instead
of in the main project:

### `business_dashboard_unrelated.html`
The original `index_7-1.html`. This is a standalone "Tandem Echo — Business
Dashboard" UI mockup. It makes zero calls to `/compare`, `/predict`,
`/generate`, or any other endpoint in this project — it's unrelated to the
Social Skills AI app and was not used as the frontend.

### `original_main.py`
The FastAPI file you uploaded directly (as opposed to the one embedded in the
notebook, which was different and larger). Its `/predict` and `/compare`
endpoints only vectorized text (`vectorizer.transform`) with **no numeric
features**, even though the trained model expects TF-IDF *plus* 10 numeric
features hstacked together. Calling the real model this way would silently
produce wrong-shaped input / wrong predictions. Superseded by
`backend/main.py` + `backend/inference.py` + `backend/feature_engineering.py`.

### `original_model_utils.py`
The `SoftVotingOnlineEnsemble` class you uploaded separately. It's identical
in logic to the class now in `backend/model_utils.py` — no functional changes,
just kept here as the original copy.

### `original_exploration_notebook.ipynb`
The full original notebook. It actually contains **two separate ML efforts**:

1. Cells 0–11: a 3-class ("A" / "B" / "Tie") model trained on
   `train_chatbot-arena.csv`. This model is never saved or used anywhere else
   in the notebook — it's exploratory and unrelated to the Social Skills AI
   product. Not carried forward.
2. Cells 12–20: the **real** Social Skills model — trained on
   `social_skills_dataset (2).csv`, saved as `social_skills_model.pkl`, and
   the one your `.pkl` files actually are. This is the pipeline
   `notebooks/training.ipynb` reproduces cleanly.

The notebook also contains a `SoftVotingOnlineEnsemble` class **redefined
inline** (not imported from a module) — which is why the original
`.pkl` files raise `AttributeError: Can't get attribute
'SoftVotingOnlineEnsemble' on <module '__main__'>` when loaded outside
Jupyter. The model was retrained (identical pipeline, identical
`random_state=42` split, identical hyperparameters) with the class imported
from `backend/model_utils.py` instead, producing a portable `.pkl`.

Other things in the original notebook that were **not** carried into the
clean app, because they're serving/experimentation concerns rather than the
core ML product:
- An in-notebook Uvicorn server (`await server.serve()`) — the backend must
  run as its own process, not from inside Jupyter.
- `/score` and `/compare` endpoints (the notebook's own version, with a
  `decisions: List[str]` request body) that call `score_decisions()` — a
  function that is **never defined** anywhere in the notebook. Calling either
  endpoint as originally written would raise `NameError`.
- `/feedback`, `/report`, human-in-the-loop `partial_fit` feedback logging,
  and `_top_contributing_terms` / `explain_choice` (LLM-based explanations).
  These are reasonable features for a v2, but they're out of scope for the
  two endpoints you specified (`/compare`, `/generate`,
  `/generate-and-compare`) and weren't wired up correctly in the source
  (e.g. `/feedback` re-saves the model on every request with no
  concurrency/versioning safeguards). Nothing was deleted — it's all still
  here in this archived notebook if you want to build on it later.
