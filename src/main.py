from data_cleaning import cleaned_data
from feature_engineering import create_features
from reading_data import load_data
from preprocess import build_preprocessor
from train import split_data, train_model, run_cross_validation
from pipeline import build_pipeline
from evaluate import evaluate_model, find_best_threshold, evaluate_with_threshold
from logger import setup_logger
from tune import tune_model
from feature_importance import get_feature_importance, show_top_features

import joblib
import os

logger = setup_logger()

try:
    logger.info("Project started")

    # Load data
    logger.info("Loading data...")
    df = load_data()
    logger.info(f"Data loaded successfully: {df.shape}")

    # Cleaning
    logger.info("Cleaning data...")
    df = cleaned_data(df)

    # Feature Engineering
    logger.info("Creating features...")
    df = create_features(df)

    # Save cleaned data
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(BASE_DIR, "Data", "Cleaned_Data.csv")
    df.to_csv(file_path, index=False)

    logger.info("Cleaned data saved successfully")

    # EDA
 
    # Features / Target
    X = df.drop(["customerID", "Churn"], axis=1)
    y = df["Churn"]

    # Split
    logger.info("Splitting train/test data...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Preprocessor
    logger.info("Building preprocessor...")
    preprocessor = build_preprocessor(X_train)

    # Pipeline
    logger.info("Creating pipeline...")
    pipeline = build_pipeline(preprocessor)

    # Cross Validation
    logger.info("Running cross validation...")
    cv_scores = run_cross_validation(
        pipeline,
        X_train,
        y_train
    )

    logger.info(
        f"Mean Cross Validation ROC-AUC: "
        f"{cv_scores.mean():.4f}"
    )

    # Hyperparameter Tuning
    logger.info("Starting hyperparameter tuning...")
    model = tune_model(
        pipeline,
        X_train,
        y_train
    )

    logger.info("Best tuned model selected.")

    preds = model.predict(X_test.head(5))
    logger.info(f"Sample Predictions: {preds}")

    # Sample Predictions
    preds = model.predict(X_test.head(5))
    logger.info(f"Sample Predictions: {preds}")

    # Default evaluation
    logger.info("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    logger.info("Evaluation completed.")

    # Threshold optimization
    logger.info("Evaluating Best Threshold...")
    best_threshold = find_best_threshold(model, X_test, y_test)

    logger.info(f"Best Threshold: {best_threshold}")

    # Optimized evaluation
    logger.info("Optimizing  Evaluation...")
    evaluate_with_threshold(model, X_test, y_test, best_threshold)

    # Feature Importance
    logger.info("Extracting feature importance...")

    importance_df = get_feature_importance(
        model,
        preprocessor,
        X_train
    )

    top_features = show_top_features(importance_df, top_n=10)


    # Save
    model_path = os.path.join(BASE_DIR, "models")
    os.makedirs(model_path, exist_ok=True)

    joblib.dump(model, os.path.join(model_path, "churn_model.pkl"))

    logger.info("Model saved successfully")

    logger.info("Customer churn pipeline executed successfully")


except Exception as e:
    logger.error(f"Project failed: {e}", exc_info=True)
