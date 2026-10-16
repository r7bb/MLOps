import json
import os

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from data import load_data, split_data

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


def fit_model(X_train, y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Return the R2 score and mean absolute error on the test set."""
    y_pred = model.predict(X_test)
    return {
        "r2_score": round(float(r2_score(y_test, y_pred)), 4),
        "mean_absolute_error": round(float(mean_absolute_error(y_test, y_pred)), 2),
    }


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "diabetes_model.pkl"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved. R2: {metrics['r2_score']}, MAE: {metrics['mean_absolute_error']}")
