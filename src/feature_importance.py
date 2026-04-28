import numpy as np
import pandas as pd
import logging

from logger import setup_logger

logger = setup_logger()


def get_feature_importance(model, preprocessor, X_train):
    """
    Extract feature importance from Logistic Regression model.
    """

    try:
        logger.info("Extracting feature importance...")

        # Get trained logistic regression model
        log_model = model.named_steps["model"]

        # Get feature names from preprocessor
        feature_names = model.named_steps["prep"].get_feature_names_out()

        # Get coefficients
        coefficients = log_model.coef_[0]

        # Create dataframe
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coefficients
        })

        # Sort by absolute impact
        importance_df["Abs_Coefficient"] = np.abs(importance_df["Coefficient"])
        importance_df = importance_df.sort_values(
            by="Abs_Coefficient",
            ascending=False
        )

        logger.info("Feature importance extraction completed.")

        return importance_df

    except Exception as e:
        logger.error(f"Error in feature importance: {e}", exc_info=True)
        raise


def show_top_features(importance_df, top_n=10):
    """
    Display top positive and negative drivers.
    """

    try:
        logger.info(f"Showing top {top_n} important features...")

        top_features = importance_df.head(top_n)

        print("\n=== TOP FEATURES DRIVING CHURN ===")
        print(top_features[["Feature", "Coefficient"]])

        logger.info("Top features displayed successfully.")

        return top_features

    except Exception as e:
        logger.error(f"Error displaying features: {e}", exc_info=True)
        raise
