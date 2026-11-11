import json
import os

import joblib
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

OUTPUT_DIR = "output"

MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}


def evaluate(model, X_train, X_test, y_train, y_test):
    """Train a model and return its R2 score and mean absolute error on the test set."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return round(float(r2_score(y_test, y_pred)), 4), round(float(mean_absolute_error(y_test, y_pred)), 2)


if __name__ == "__main__":
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    print(f"{'Model':<20}{'R2':>8}{'MAE':>8}")
    for name, model in MODELS.items():
        r2, mae = evaluate(model, X_train, X_test, y_train, y_test)
        results[name] = {"r2_score": r2, "mean_absolute_error": mae}
        print(f"{name:<20}{r2:>8.4f}{mae:>8.2f}")

    best_name = max(results, key=lambda n: results[n]["r2_score"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(MODELS[best_name], os.path.join(OUTPUT_DIR, "diabetes_model.pkl"))
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    print(f"\nBest model: {best_name}")
    print(f"Saved to {OUTPUT_DIR}/diabetes_model.pkl")
    print("The model training was successful")
