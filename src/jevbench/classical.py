"""Training-only TF-IDF vocabulary; three-candidate full-data validation budget.

Selected models remain fitted only on train. Validation is never merged into train,
so labeled training budgets can be matched exactly with the LoRA arm. Equal-label
runs fix C=1 / alpha=1 a priori and do not use validation labels. LinearSVC
returns no probabilities; decision margins are not treated as probabilities.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from .data import select_train_rows, validate_prepared
from .types import Prediction, PreparedDataset

CLASSICAL_MODELS = ("majority", "logistic_regression", "linear_svc", "multinomial_nb")
ALIASES = {"dummy": "majority", "dummy_majority": "majority", "logreg": "logistic_regression",
           "tfidf_logreg": "logistic_regression", "svm": "linear_svc", "tfidf_svc": "linear_svc",
           "nb": "multinomial_nb", "tfidf_nb": "multinomial_nb"}


def _make_vectorizer() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                max_features=60000, dtype=np.float64)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                                max_features=80000, dtype=np.float64)),
    ])


def _candidates(name: str, seed: int):
    if name == "majority":
        return [({"strategy": "most_frequent"}, DummyClassifier(strategy="most_frequent"))]
    if name == "logistic_regression":
        return [({"C": c}, LogisticRegression(C=c, max_iter=2000, solver="lbfgs", random_state=seed))
                for c in (0.1, 1.0, 10.0)]
    if name == "linear_svc":
        return [({"C": c}, LinearSVC(C=c, max_iter=5000, random_state=seed, dual="auto"))
                for c in (0.1, 1.0, 10.0)]
    if name == "multinomial_nb":
        return [({"alpha": alpha}, MultinomialNB(alpha=alpha)) for alpha in (0.1, 1.0, 10.0)]
    raise ValueError(f"Unknown classical model {name!r}; choose from {CLASSICAL_MODELS}")


def fit_classical(name: str, dataset: PreparedDataset, seed: int = 42,
                  train_per_class: int | None = None) -> tuple[list[Prediction], dict[str, Any]]:
    name = ALIASES.get(name, name)
    candidates = _candidates(name, seed)
    # Equal-label track cannot spend extra validation labels on model selection.
    tune_on_validation = train_per_class is None
    if not tune_on_validation and name != "majority":
        candidates = [candidates[1]]  # preregistered C=1 or alpha=1
    validate_prepared(dataset)
    train = select_train_rows(dataset.train, train_per_class, seed)
    labels = list(range(len(dataset.labels)))
    if {row.label for row in train} != set(labels):
        raise ValueError("Training subset must represent all dataset classes")
    start = time.perf_counter()
    vectorizer = None if name == "majority" else _make_vectorizer()
    if vectorizer is None:
        x_train = np.zeros((len(train), 1))
        x_val = np.zeros((len(dataset.validation), 1)) if tune_on_validation else None
    else:
        x_train = vectorizer.fit_transform([row.text for row in train])
        x_val = vectorizer.transform([row.text for row in dataset.validation]) if tune_on_validation else None
    feature_fit_s = time.perf_counter() - start
    y_train = [row.label for row in train]
    y_val = [row.label for row in dataset.validation]
    trials = []
    best_score, selected, selected_params = -float("inf"), None, None
    for params, estimator in candidates:
        candidate_start = time.perf_counter()
        estimator.fit(x_train, y_train)
        fit_s = time.perf_counter() - candidate_start
        score = None
        if tune_on_validation:
            val_predictions = estimator.predict(x_val)
            score = float(f1_score(y_val, val_predictions, labels=labels, average="macro", zero_division=0))
        trials.append({"parameters": params, "validation_macro_f1": score, "fit_s": fit_s})
        if selected is None or (score is not None and score > best_score):  # first-candidate tiebreak
            best_score, selected, selected_params = score, estimator, params
    fit_total_s = time.perf_counter() - start
    assert selected is not None
    # Test text is first transformed only after every validation decision is frozen.
    inference_start = time.perf_counter()
    x_test = np.zeros((len(dataset.test), 1)) if vectorizer is None else vectorizer.transform([row.text for row in dataset.test])
    predictions = selected.predict(x_test)
    probabilities = selected.predict_proba(x_test) if hasattr(selected, "predict_proba") else None
    inference_s = time.perf_counter() - inference_start
    classes = list(selected.classes_)
    ordered_probabilities = None if probabilities is None else probabilities[:, [classes.index(label) for label in labels]]
    result = [Prediction(
        row_id=row.id, label=int(predictions[i]),
        probabilities=None if ordered_probabilities is None else ordered_probabilities[i].astype(float).tolist(),
        latency_s=inference_s / len(dataset.test),
        metadata={"latency_measurement": "amortized batch inference including TF-IDF transform"},
    ) for i, row in enumerate(dataset.test)]
    metadata: dict[str, Any] = {
        "model": name, "seed": seed, "sklearn_version": sklearn_version,
        "training_rows": len(train), "train_per_class": train_per_class,
        "training_row_ids": [row.id for row in train],
        "training_row_ids_sha256": hashlib.sha256(json.dumps([row.id for row in train], separators=(",", ":")).encode()).hexdigest(),
        "validation_rows": len(dataset.validation) if tune_on_validation else 0, "test_rows": len(dataset.test),
        "selection_metric": "validation_macro_f1" if tune_on_validation else "fixed_a_priori", "selected_parameters": selected_params,
        "selected_validation_macro_f1": best_score, "hyperparameter_trials": trials,
        "candidate_budget": len(candidates), "feature_fit_s": feature_fit_s,
        "fit_and_selection_s": fit_total_s, "test_inference_s": inference_s,
        "refit_with_validation": False, "probability_kind": "none" if probabilities is None else "native_uncalibrated",
        "latency_measurement": "amortized batch; not individual request latency",
        "features": None if vectorizer is None else {
            "word_ngrams": [1, 2], "char_wb_ngrams": [3, 5], "sublinear_tf": True,
            "max_word_features": 60000, "max_char_features": 80000,
            "vocabulary_fit_split": "train only", "feature_count": x_train.shape[1],
        },
    }
    return result, metadata
