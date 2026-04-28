import logging

from sklearn.model_selection import GridSearchCV


def tune_model(pipeline, X_train, y_train):
    """
    Tune Logistic Regression pipeline using GridSearchCV.
    """

    try:
        logging.info("Starting hyperparameter tuning...")

        param_grid = {
            "model__C": [0.01, 0.1, 1, 10],
            "model__class_weight": [None, "balanced"]
        }

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_train)

        logging.info("Hyperparameter tuning completed.")
        logging.info(f"Best Score: {grid.best_score_:.4f}")
        logging.info(f"Best Params: {grid.best_params_}")

        return grid.best_estimator_

    except Exception:
        logging.exception("Error during hyperparameter tuning")
        raise
