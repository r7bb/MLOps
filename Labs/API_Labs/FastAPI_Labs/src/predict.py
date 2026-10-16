import os

import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "diabetes_model.pkl")
model = joblib.load(MODEL_PATH)


def predict_data(X):
    """Predict diabetes progression scores for the input data."""
    return model.predict(X)
