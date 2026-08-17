"""
Shared feature-engineering logic for the Social Skills AI model.

This module is the SINGLE SOURCE OF TRUTH for turning
(category, situation, prompt, response_a, response_b) into the exact
feature matrix the model expects: TF-IDF over a combined text field,
hstacked with 10 scaled numeric features.

It is used by BOTH:
  - notebooks/training.ipynb (fitting the vectorizer/scaler/model)
  - backend/main.py (inference at request time)

so that training-time and inference-time feature construction can never
drift apart.

Feature layout (must match training exactly - see notebooks/training.ipynb):

Text:
    combined_text = "CATEGORY: {category} SITUATION: {situation} "
                     "PROMPT: {prompt} RESPONSE_A: {response_a} "
                     "RESPONSE_B: {response_b}"
    -> TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)

Numeric (computed SEPARATELY for response_a and response_b - never by
splitting stats of a single combined string):
    char_len_a, word_count_a, has_question_a, has_exclaim_a,
    char_len_b, word_count_b, has_question_b, has_exclaim_b,
    length_diff_abs, a_is_longer
    -> MaxAbsScaler

Final feature matrix = hstack([tfidf_matrix, scaled_numeric]).tocsr()
"""

from typing import Iterable, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp


NUMERIC_FEATURE_COLUMNS = [
    "char_len_a",
    "word_count_a",
    "has_question_a",
    "has_exclaim_a",
    "char_len_b",
    "word_count_b",
    "has_question_b",
    "has_exclaim_b",
    "length_diff_abs",
    "a_is_longer",
]


def text_stats(series: pd.Series) -> pd.DataFrame:
    """Per-string statistics: char_len, word_count, has_question, has_exclaim."""
    s = series.astype(str)
    return pd.DataFrame({
        "char_len": s.str.len(),
        "word_count": s.str.split().apply(len),
        "has_question": s.str.contains(r"\?").astype(int),
        "has_exclaim": s.str.contains(r"!").astype(int),
    })


def build_numeric_features(resp_a: pd.Series, resp_b: pd.Series) -> pd.DataFrame:
    """
    Build the 10 numeric features from response_a and response_b.

    CRITICAL: stats for A and stats for B are each computed from their own
    text, independently - never from a shared/combined string. Only after
    both are computed separately do we derive the two comparison features
    (length_diff_abs, a_is_longer).
    """
    stats_a = text_stats(resp_a).add_suffix("_a")
    stats_b = text_stats(resp_b).add_suffix("_b")
    numeric = pd.concat([stats_a.reset_index(drop=True), stats_b.reset_index(drop=True)], axis=1)

    raw_diff = stats_a["char_len_a"].reset_index(drop=True) - stats_b["char_len_b"].reset_index(drop=True)
    numeric["length_diff_abs"] = raw_diff.abs()
    numeric["a_is_longer"] = (raw_diff > 0).astype(int)

    return numeric[NUMERIC_FEATURE_COLUMNS]


def build_combined_text(
    category: Union[str, Iterable[str]],
    situation: Union[str, Iterable[str]],
    prompt: Union[str, Iterable[str]],
    response_a: Union[str, Iterable[str]],
    response_b: Union[str, Iterable[str]],
) -> pd.Series:
    """Build the combined text field the TF-IDF vectorizer was fit on."""
    category = pd.Series(category).astype(str)
    situation = pd.Series(situation).astype(str)
    prompt = pd.Series(prompt).astype(str)
    response_a = pd.Series(response_a).astype(str)
    response_b = pd.Series(response_b).astype(str)

    n = max(len(category), len(situation), len(prompt), len(response_a), len(response_b))
    if len(category) == 1 and n > 1:
        category = pd.Series([category.iloc[0]] * n)
    if len(situation) == 1 and n > 1:
        situation = pd.Series([situation.iloc[0]] * n)

    return (
        "CATEGORY: " + category.reset_index(drop=True)
        + " SITUATION: " + situation.reset_index(drop=True)
        + " PROMPT: " + prompt.reset_index(drop=True)
        + " RESPONSE_A: " + response_a.reset_index(drop=True)
        + " RESPONSE_B: " + response_b.reset_index(drop=True)
    )


def make_features(
    vectorizer,
    scaler,
    prompt,
    response_a,
    response_b,
    category="General",
    situation="",
):
    """
    Build the full inference-time feature matrix for one or more examples,
    using an ALREADY-FITTED vectorizer and scaler (transform only - never
    fit again at inference time).

    Accepts either single strings or equal-length iterables for
    prompt/response_a/response_b (category/situation may be a single string
    applied to all rows, or per-row iterables).

    Returns a scipy.sparse.csr_matrix ready to feed into the model.
    """
    combined_text = build_combined_text(category, situation, prompt, response_a, response_b)

    tfidf_part = vectorizer.transform(combined_text)

    numeric = build_numeric_features(
        pd.Series(response_a).astype(str).reset_index(drop=True),
        pd.Series(response_b).astype(str).reset_index(drop=True),
    )
    numeric_scaled = scaler.transform(numeric)

    return sp.hstack([tfidf_part, numeric_scaled]).tocsr()
