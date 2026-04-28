import logging
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def build_pipeline(preprocessor):
    try:
        logging.info("Starting pipeline creation process")

        logging.info("Initializing Logistic Regression model")
        logging.info(
            "Model parameters: max_iter=2000, class_weight='balanced'"
        )

        pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            ))
        ])

        logging.info("Pipeline created successfully")
        logging.info(f"Pipeline steps: {pipeline.named_steps.keys()}")

        return pipeline

    except Exception as e:
        logging.exception("Error occurred while building pipeline")
        raise
