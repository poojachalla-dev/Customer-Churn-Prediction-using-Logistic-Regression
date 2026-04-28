import logging
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(df):
    try:
        logging.info("Starting EDA")

        # Basic Info
        logging.info(f"Dataset shape: {df.shape}")
        logging.info(f"Missing values:\n{df.isnull().sum()}")
        logging.info(
            f"Churn distribution: "
            f"{df['Churn'].value_counts().to_dict()}"
        )

        logging.info("Calculating descriptive statistics")
        summary = df.describe(include="all")

        # -------------------------
        # Visualizations
        # -------------------------

        logging.info("Generating Customer Churn Count plot")
        plt.figure(figsize=(6, 4))
        sns.countplot(x="Churn", data=df)
        plt.title("Customer Churn Count")
        plt.show()

        logging.info("Generating Tenure Distribution plot")
        plt.figure(figsize=(6, 4))
        sns.histplot(df["tenure"], bins=30, kde=True)
        plt.title("Tenure Distribution")
        plt.show()

        logging.info("Generating Monthly Charges by Churn plot")
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
        plt.title("Monthly Charges by Churn")
        plt.show()

        logging.info("Generating Contract Type vs Churn plot")
        plt.figure(figsize=(8, 4))
        sns.countplot(x="Contract", hue="Churn", data=df)
        plt.title("Contract Type vs Churn")
        plt.show()

        logging.info("EDA completed successfully")

        return summary

    except Exception:
        logging.exception("EDA failed")
        raise
