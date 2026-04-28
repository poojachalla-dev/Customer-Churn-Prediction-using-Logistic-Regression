import pandas as pd
import numpy as np
import os
import logging

def load_data():
    try:
        logging.info("Starting data loading process")

        BASE_DIR = os.path.dirname(os.path.dirname(__file__))
        logging.info(f"Base directory identified: {BASE_DIR}")

        file_name = os.path.join(
            BASE_DIR,
            "Data",
            "Telco_Customer_Churn_Dataset.csv"
        )

        logging.info(f"File path created: {file_name}")

        df = pd.read_csv(file_name)

        logging.info(f"Dataset loaded successfully with shape: {df.shape}")

        return df

    except Exception as e:
        logging.exception("Error occurred while loading dataset")
        raise
