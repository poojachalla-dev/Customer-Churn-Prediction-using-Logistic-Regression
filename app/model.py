import joblib
import logging

MODEL_PATH = "models/churn_model.pkl"

def load_model():
    try:
        logging.info(f"Loading model from: {MODEL_PATH}")

        model = joblib.load(MODEL_PATH)

        logging.info("Model loaded successfully")

        return model

    except Exception:
        logging.exception("Failed to load model")
        raise

