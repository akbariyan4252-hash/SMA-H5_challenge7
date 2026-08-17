"""
Custom estimator used by the Social Skills AI model.

This class is preserved EXACTLY as it was defined in the original training
notebook (Decision_System_Challenge7_final_4_new_ui_10_.ipynb, cell 15) and in
the previously-provided model_utils.py.

It must live in a normal, importable module (this file) rather than inside
a notebook's `__main__` namespace. If it is redefined inside `__main__`
(e.g. directly in a notebook or a script run as `python script.py`), joblib
will pickle it as `__main__.SoftVotingOnlineEnsemble`, and loading it from
anywhere else (like `uvicorn backend.main:app`) will fail with:

    AttributeError: Can't get attribute 'SoftVotingOnlineEnsemble'
    on <module '__main__'>

By keeping the class here and always importing it as
`from backend.model_utils import SoftVotingOnlineEnsemble` (both when
training/saving AND when loading in the API), the pickled model is portable
within this project.
"""

import numpy as np


class SoftVotingOnlineEnsemble:
    """A simple soft-voting ensemble of online (partial_fit-capable) classifiers."""

    def __init__(self, estimators):
        self.estimators = estimators
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        for _, est in self.estimators:
            est.fit(X, y)

        return self

    def partial_fit(self, X, y, classes=None):
        if self.classes_ is None:
            self.classes_ = (
                np.array(classes)
                if classes is not None
                else np.unique(y)
            )

        for _, est in self.estimators:
            est.partial_fit(
                X,
                y,
                classes=self.classes_
            )

        return self

    def predict_proba(self, X):
        probs = [
            est.predict_proba(X)
            for _, est in self.estimators
        ]

        return np.mean(probs, axis=0)  # soft voting

    def predict(self, X):
        avg_proba = self.predict_proba(X)

        return self.classes_[
            np.argmax(avg_proba, axis=1)
        ]
