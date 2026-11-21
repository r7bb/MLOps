import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

EXPERIMENT_NAME = "Diabetes ElasticNet"
ALPHAS = [0.001, 0.01, 0.1]
L1_RATIOS = [0.2, 0.5, 0.8]


def eval_metrics(actual, pred):
    """Return RMSE, MAE and R2 for the predictions."""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def plot_predictions(actual, pred, title):
    """Create a predicted vs actual scatter plot."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(actual, pred, alpha=0.6)
    limits = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    ax.plot(limits, limits, "r--")
    ax.set_xlabel("Actual progression")
    ax.set_ylabel("Predicted progression")
    ax.set_title(title)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    best_name, best_r2 = None, float("-inf")
    for alpha in ALPHAS:
        for l1_ratio in L1_RATIOS:
            run_name = f"alpha={alpha}_l1={l1_ratio}"
            with mlflow.start_run(run_name=run_name):
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
                model.fit(train_x, train_y)
                pred = model.predict(test_x)
                rmse, mae, r2 = eval_metrics(test_y, pred)

                mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
                mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
                mlflow.set_tag("dataset", "scikit-learn diabetes")

                fig = plot_predictions(test_y, pred, run_name)
                mlflow.log_figure(fig, "predicted_vs_actual.png")
                plt.close(fig)

                signature = infer_signature(train_x, model.predict(train_x))
                mlflow.sklearn.log_model(model, "model", signature=signature)

                print(f"{run_name:<22} RMSE: {rmse:.2f}  MAE: {mae:.2f}  R2: {r2:.4f}")
                if r2 > best_r2:
                    best_name, best_r2 = run_name, r2

    print(f"\nBest run: {best_name} with R2 {best_r2:.4f}")
    print("Open the MLflow UI to compare all runs")
