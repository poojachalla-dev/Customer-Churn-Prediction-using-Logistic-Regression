from fastapi import FastAPI, HTTPException
from src.feature_engineering import create_features
import pandas as pd
import logging

from app.model import load_model
from app.schemas import CustomerInput

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App metadata
app = FastAPI(
    title="Churn Prediction API",
    description="Production-ready ML API for customer churn prediction",
    version="1.0.0"
)

# Load model once on startup
model = load_model()
logger.info("Model loaded successfully")

@app.get("/")
def home():
    return {
        "message": "Churn Prediction API Running",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.post("/predict")
def predict(data: CustomerInput):
    try:
        logger.info("Prediction request received")

        input_df = pd.DataFrame([data.dict()])

        input_df = create_features(input_df)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        result = {
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
            "risk_level": "High" if probability > 0.7 else "Low"
        }

        logger.info(f"Prediction completed: {result}")

        return result

    except Exception as e:
        logging.exception("Prediction error")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

