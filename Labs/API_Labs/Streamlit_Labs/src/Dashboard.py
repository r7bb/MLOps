import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from streamlit.logger import get_logger

FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"
FASTAPI_MODEL_LOCATION = Path(__file__).resolve().parents[2] / "FastAPI_Labs" / "model" / "diabetes_model.pkl"
LOGGER = get_logger(__name__)

SCORE_MIN, SCORE_MAX = 25, 346

FEATURES = {
    "age": ("Age", 0.0381),
    "sex": ("Sex", 0.0507),
    "bmi": ("Body mass index", 0.0617),
    "bp": ("Average blood pressure", 0.0219),
    "s1": ("Total serum cholesterol", -0.0442),
    "s2": ("Low density lipoproteins", -0.0348),
    "s3": ("High density lipoproteins", -0.0434),
    "s4": ("Total cholesterol to HDL ratio", -0.0026),
    "s5": ("Log of serum triglycerides", 0.0199),
    "s6": ("Blood sugar level", -0.0176),
}


def progression_level(score):
    """Group the score into Low, Moderate or High based on the dataset range."""
    if score < 100:
        return "Low"
    if score <= 180:
        return "Moderate"
    return "High"


def show_backend_status():
    """Check the FastAPI backend and show the model's test metrics."""
    try:
        response = requests.get(FASTAPI_BACKEND_ENDPOINT, timeout=5)
        if response.status_code != 200:
            st.warning("Problem connecting 😭")
            return
        st.success("Backend online ✅")
    except requests.ConnectionError as ce:
        LOGGER.error(f"Backend error: {ce}")
        st.error("Backend offline 😱")
        return

    try:
        info = requests.get(f"{FASTAPI_BACKEND_ENDPOINT}/model_info", timeout=5).json()
        metrics = info["test_metrics"]
        col1, col2 = st.columns(2)
        col1.metric("Test R2", metrics["r2_score"])
        col2.metric("Test MAE", metrics["mean_absolute_error"])
    except Exception as e:
        LOGGER.error(f"Model info error: {e}")
        st.warning("Could not load model metrics")


def show_prediction(client_input):
    """Send the input to the backend and show the predicted score."""
    try:
        with st.spinner("Predicting..."):
            response = requests.post(f"{FASTAPI_BACKEND_ENDPOINT}/predict", json=client_input, timeout=10)

        if response.status_code != 200:
            st.toast(f":red[Status: {response.status_code}. Check backend]", icon="🔴")
            return

        score = response.json()["predicted_progression"]
        level = progression_level(score)

        st.metric("Predicted progression score", score)
        st.progress(min(max((score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN), 0.0), 1.0))
        st.caption(f"Scores in the dataset range from {SCORE_MIN} to {SCORE_MAX}")

        if level == "High":
            st.error(f"Progression level: {level}")
        elif level == "Moderate":
            st.warning(f"Progression level: {level}")
        else:
            st.success(f"Progression level: {level}")

        st.write("#### Input values")
        st.bar_chart(pd.DataFrame.from_dict(client_input, orient="index", columns=["value"]))
    except Exception as e:
        st.toast(":red[Backend error. Refresh and retry]", icon="🔴")
        LOGGER.error(f"Prediction error: {e}")


def run():
    st.set_page_config(page_title="Diabetes Progression Dashboard", page_icon="🩺")

    with st.sidebar:
        show_backend_status()

        st.info("Configure patient values")
        slider_values = {}
        for key, (label, default) in FEATURES.items():
            slider_values[key] = st.slider(label, -0.2, 0.2, default, 0.0001, format="%.4f", key=key)

        test_input_file = st.file_uploader("Upload test prediction file", type=["json"])
        if test_input_file:
            st.write("Preview file")
            test_input_data = json.load(test_input_file)
            st.json(test_input_data)
            st.session_state["IS_JSON_FILE_AVAILABLE"] = True
            st.session_state["test_input_data"] = test_input_data
        else:
            st.session_state["IS_JSON_FILE_AVAILABLE"] = False

        predict_button = st.button("Predict")

    st.write("# Diabetes Progression Prediction 🩺")
    st.write("Adjust the patient values in the sidebar or upload a JSON file, then click **Predict**.")
    st.caption("Built for an MLOps lab. Not medical advice.")

    if predict_button:
        if FASTAPI_MODEL_LOCATION.is_file():
            if st.session_state.get("IS_JSON_FILE_AVAILABLE"):
                client_input = st.session_state["test_input_data"]["input_test"]
            else:
                client_input = slider_values
            show_prediction(client_input)
        else:
            LOGGER.warning("diabetes_model.pkl not found")
            st.toast(":red[Model not found. Run train.py in FastAPI_Labs]", icon="🔥")


if __name__ == "__main__":
    run()
