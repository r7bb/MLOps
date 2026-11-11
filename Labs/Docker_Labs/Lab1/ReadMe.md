# Docker Lab 1: Diabetes Model Comparison in a Container

Trains and compares three regression models on the scikit-learn diabetes dataset inside a **Docker** container, then saves the best model to the host machine through a volume.

Based on [Docker Lab 1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Docker_Labs/Lab1) from Prof. Ramin Mohammadi's MLOps repo.

## Modifications from Original

| Original Lab | This Lab |
|---|---|
| Iris dataset (classification) | Diabetes dataset (regression) |
| One model (Random Forest Classifier) | Three models compared: Linear Regression, Ridge, Random Forest |
| Model saved inside the container and lost when it stops | Best model saved to the host through a volume mount |
| Prints a single success message | Prints a comparison table with R2 and MAE for each model |
| No metrics saved | `metrics.json` with every model's scores and the best model |
| `python:3.10` base image | Smaller `python:3.11-slim` base image |
| Code and packages copied together | Requirements installed first, so code changes rebuild faster |
| Container kept after it finishes | Removed automatically with `--rm` |

## Build and Run

```bash
cd Labs/Docker_Labs/Lab1
docker build -t lab1:v1 .
docker run --rm -v "$(pwd)/output:/app/output" lab1:v1
```

The trained model and metrics appear in `output/` on your machine.

## Save the Image

```bash
docker save lab1:v1 > my_image.tar
```

`my_image.tar` and `output/` are listed in this folder's `.gitignore`, so they stay out of the repo.

## Results

| Model | Test R2 | Test MAE |
|---|---|---|
| Linear Regression | 0.4526 | 42.79 |
| Ridge Regression | **0.4609** | 43.00 |
| Random Forest | 0.4428 | 44.05 |

Ridge Regression has the highest R2, so it is saved as the best model.

## Screenshots

![Docker run output](assets/docker_run.png)

![Docker image](assets/docker_images.png)

## Project Structure

```
Lab1/
├── assets/
│   ├── docker_images.png
│   └── docker_run.png
├── src/
│   ├── main.py
│   └── requirements.txt
├── .gitignore
├── dockerfile
└── ReadMe.md
```
