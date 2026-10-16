Heart Disease Prediction — Deployment Files

Files:
- app.py: Streamlit interface
- requirements.txt: required Python packages
- heart_disease_model.pkl: must be generated from your trained Task 2 notebook

IMPORTANT:
A trained .pkl model cannot be correctly created until the notebook has been run and the model trained.
In your Task 2 notebook, after model.fit(...), save the trained model/pipeline using:

import joblib
joblib.dump(
    {"model": model, "features": ["age", "sex", "cp", "thalach"]},
    "heart_disease_model.pkl"
)

Then download heart_disease_model.pkl from the Colab Files panel and place it beside app.py.
Upload all three files to GitHub, then deploy app.py using Streamlit Community Cloud.
