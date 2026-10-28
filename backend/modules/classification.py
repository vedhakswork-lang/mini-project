from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def run_classifier(X: np.ndarray, y: list[str] | np.ndarray, model_name: str = "lda", test_size: float = 0.25, random_state: int = 42) -> dict[str, Any]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("Feature matrix and target labels must have the same number of rows.")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("At least two classes are required for classification.")
    if np.min(counts) < 2:
        raise ValueError("Each class needs at least two examples for a train/test split.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    if model_name.lower() == "lda":
        model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    elif model_name.lower() == "svm":
        model = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=False, random_state=random_state))
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "model_name": model_name,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "classes": classes.tolist(),
        "y_test": y_test.tolist(),
        "y_pred": predictions.tolist(),
        "model": model,
    }
