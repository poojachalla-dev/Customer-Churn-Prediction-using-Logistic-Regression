from fastapi import FastAPI 
import pandas as pd

from app.model import load_model
from app.schemas import CustomerInput

app = FastAPI(title="Churn Prediction API")

model = load_model()

@app.post("/predict")
def predict(data: CustomerInput):
    input_df = pd.DataFrame([data.dict()])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability)

    }