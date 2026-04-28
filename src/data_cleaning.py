import pandas as pd
import logging

def cleaned_data(df):
    try:
        logging.info("Starting data cleaning process")

        logging.info("Converting TotalCharges to numeric datatype")
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

        missing_values = df["TotalCharges"].isnull().sum()
        logging.info(f"Missing values found in TotalCharges: {missing_values}")

        logging.info("Dropping rows with missing TotalCharges")
        df.dropna(subset=["TotalCharges"], inplace=True)

        logging.info("Mapping Churn column values: Yes=1, No=0")
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

        churn_counts = df["Churn"].value_counts().to_dict()
        logging.info(f"Churn value counts after mapping: {churn_counts}")

        logging.info(f"Data cleaning completed successfully. Final shape: {df.shape}")

        return df

    except Exception as e:
        logging.exception("Error occurred during data cleaning")
        raise
