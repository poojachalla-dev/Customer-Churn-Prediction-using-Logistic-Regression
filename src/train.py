import logging
from sklearn.model_selection import train_test_split


def split_data(X, y):
    try:
        logging.info("Starting train-test split process")

        logging.info(
            "Splitting dataset with test_size=0.2, "
            "random_state=42, stratify=y"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        logging.info("Train-test split completed successfully")
        logging.info(f"X_train shape: {X_train.shape}")
        logging.info(f"X_test shape: {X_test.shape}")
        logging.info(f"y_train shape: {y_train.shape}")
        logging.info(f"y_test shape: {y_test.shape}")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.exception("Error occurred during data splitting")
        raise


def train_model(pipeline, X_train, y_train):
    try:
        logging.info("Starting model training process")

        logging.info("Fitting pipeline on training data")
        pipeline.fit(X_train, y_train)

        logging.info("Model training completed successfully")

        model = pipeline.named_steps["model"]

        logging.info(
            f"Trained model type: {type(model).__name__}"
        )

        logging.info(
            f"Pipeline steps after training: "
            f"{list(pipeline.named_steps.keys())}"
        )

        return pipeline

    except Exception as e:
        logging.exception("Error occurred during mo


