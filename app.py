import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("Heart Disease Prediction System")
st.warning(
    "Educational demonstration only. This model is not a medical diagnostic tool "
    "and must not be used to make health decisions."
)

model_path = Path(__file__).with_name("heart_disease_model.pkl")
if not model_path.exists():
    st.error(
        "Model file not found. Run the Task 2 Jupyter notebook, save the trained "
        "model as heart_disease_model.pkl, and upload it beside app.py."
    )
    st.stop()

artifact = joblib.load(model_path)

# Support either a dictionary containing model/features or a directly saved sklearn pipeline.
if isinstance(artifact, dict) and "model" in artifact:
    model = artifact["model"]
    feature_names = artifact.get("features", ["age", "sex", "cp", "thalach"])
else:
    model = artifact
    feature_names = ["age", "sex", "cp", "thalach"]

st.write("Enter the four input attributes used by the model.")

age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex code (UCI dataset: 1 = male, 0 = female)", options=[1, 0])
cp = st.selectbox("Chest-pain type (cp)", options=[1, 2, 3, 4])
thalach = st.number_input(
    "Maximum heart rate achieved (thalach)",
    min_value=50, max_value=250, value=150
)

if st.button("Predict"):
    patient = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "thalach": thalach
    }], columns=feature_names)

    prediction = int(model.predict(patient)[0])
    probability = float(model.predict_proba(patient)[0, 1])

    if prediction == 1:
        st.subheader("Model output: Disease-present class")
    else:
        st.subheader("Model output: No-disease class")

    st.write(f"Model-estimated probability for class 1: {probability:.1%}")
    st.caption(
        "For coursework demonstration only. Consult a qualified healthcare "
        "professional for medical concerns."
    )
