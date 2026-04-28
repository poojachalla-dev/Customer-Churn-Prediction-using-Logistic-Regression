import logging

def create_features(df):
    try:
        logging.info("Starting feature engineering process")

        logging.info("Creating AvgMonthlySpend feature")
        df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

        logging.info("Creating IsNewCustomer feature")
        df["IsNewCustomer"] = (df["tenure"] < 12).astype(int)

        logging.info("Calculating MonthlyCharges median")
        monthly_median = df["MonthlyCharges"].median()
        logging.info(f"MonthlyCharges median value: {monthly_median}")

        logging.info("Creating HighValueCustomer feature")
        df["HighValueCustomer"] = (
            df["MonthlyCharges"] > monthly_median
        ).astype(int)

        logging.info("Feature creation completed successfully")

        logging.info(
            f"AvgMonthlySpend sample values: "
            f"{df['AvgMonthlySpend'].head().tolist()}"
        )

        logging.info(
            f"IsNewCustomer distribution: "
            f"{df['IsNewCustomer'].value_counts().to_dict()}"
        )

        logging.info(
            f"HighValueCustomer distribution: "
            f"{df['HighValueCustomer'].value_counts().to_dict()}"
        )

        logging.info(f"Updated dataset shape after feature engineering: {df.shape}")

        return df

    except Exception as e:
        logging.exception("Error occurred during feature engineering")
        raise
