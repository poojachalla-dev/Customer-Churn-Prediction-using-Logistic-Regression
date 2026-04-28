import logging
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)


def split_data(X, y):
    """
    Split dataset into train and test sets.
    """

    try:
        logging.info("Starting train-test split")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        logging.info("Train-test split completed")
        logging.info(f"X_train shape: {X_train.shape}")
        logging.info(f"X_test shape: {X_test.shape}")
        logging.info(f"y_train shape: {y_train.shape}")
        logging.info(f"y_test shape: {y_test.shape}")

        return X_train, X_test, y_train, y_test

    except Exception:
        logging.exception("Error during train-test split")
        raise


def run_cross_validation(pipeline, X_train, y_train):
    """
    Perform 5-fold cross validation.
    """

    try:
        logging.info("Starting cross validation")

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="roc_auc"
        )

        logging.info(f"Cross-validation scores: {scores}")
        logging.info(f"Mean ROC-AUC: {np.mean(scores):.4f}")

        return scores

    except Exception:
        logging.exception("Error during cross validation")
        raise


def train_model(pipeline, X_train, y_train):
    """
    Train pipeline on training data.
    """

    try:
        logging.info("Starting model training")

        pipeline.fit(X_train, y_train)

        logging.info("Model training completed")

        model = pipeline.named_steps["model"]

        logging.info(
            f"Model type: {type(model).__name__}"
        )

        return pipeline

    except Exception:
        logging.exception("Error during model training")
        raise
