import numpy as np


class SoftVotingOnlineEnsemble:

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

        return np.mean(probs, axis=0)

    def predict(self, X):
        avg_proba = self.predict_proba(X)

        return self.classes_[
            np.argmax(avg_proba, axis=1)
        ]
