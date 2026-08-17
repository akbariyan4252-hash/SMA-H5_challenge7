"""
Tests for the ML core (model + feature engineering) that don't require
FastAPI to be installed. Run with:

    python -m pytest tests/test_inference.py -v

or, without pytest:

    python tests/test_inference.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.inference import compare_options  # noqa: E402
from backend.feature_engineering import build_numeric_features  # noqa: E402
import pandas as pd  # noqa: E402


def test_model_artifacts_load():
    # Importing backend.inference already loads model/vectorizer/scaler;
    # if that succeeded, this test passes.
    from backend import inference
    assert inference._model is not None
    assert inference._vectorizer is not None
    assert inference._scaler is not None


def test_compare_returns_winner_and_probabilities():
    result = compare_options(
        "I want to start a conversation with a new person.",
        "Ask about their day and then introduce myself.",
        "Ask how they are and then introduce myself.",
    )
    assert result["winner"] in ("Response A", "Response B")
    assert result["prediction"] in (0, 1)
    assert 0 <= result["response_a_probability"] <= 100
    assert 0 <= result["response_b_probability"] <= 100
    total = result["response_a_probability"] + result["response_b_probability"]
    assert abs(total - 100) < 0.5


def test_numeric_features_computed_separately_for_a_and_b():
    # Response A and B must never share stats - this guards against the
    # original bug where both were derived from one combined string.
    a = pd.Series(["Short."])
    b = pd.Series(["This is a much longer response with more words in it!"])
    numeric = build_numeric_features(a, b)
    row = numeric.iloc[0]
    assert row["char_len_a"] != row["char_len_b"]
    assert row["word_count_a"] != row["word_count_b"]
    assert row["has_exclaim_a"] == 0
    assert row["has_exclaim_b"] == 1
    assert row["a_is_longer"] == 0  # B is longer
    assert row["length_diff_abs"] > 0


def test_compare_is_order_sensitive():
    # Swapping A and B should generally swap the recommendation direction -
    # a regression guard against features being computed identically for
    # both slots.
    prompt = "I want to start a conversation with a new person."
    a = "Ask about their day and then introduce myself."
    b = "Ask how they are and then introduce myself."

    r1 = compare_options(prompt, a, b)
    r2 = compare_options(prompt, b, a)

    # The probability the model assigns to "the practical option" shouldn't
    # be identical when it moves from slot A to slot B.
    assert r1["response_a_probability"] != r2["response_b_probability"] or \
           r1["response_b_probability"] != r2["response_a_probability"]


if __name__ == "__main__":
    test_model_artifacts_load()
    test_compare_returns_winner_and_probabilities()
    test_numeric_features_computed_separately_for_a_and_b()
    test_compare_is_order_sensitive()
    print("All inference tests passed.")
