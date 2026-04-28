from data_cleaning import cleaned_data
from feature_engineering import create_features
from reading_data import load_data
from eda import run_eda
from preprocess import build_preprocessor
from train import split_data, train_model
from pipeline import build_pipeline
from evaluate import evaluate_model
from logger import setup_logger

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
    logger.info("Running EDA...")
    summary = run_eda(df)

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

    # Training
    logger.info("Training model...")
    model = train_model(pipeline, X_train, y_train)

    preds = model.predict(X_test.head(5))
    logger.info(f"Sample Predictions: {preds}")

    # Evaluate
    logger.info("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    logger.info("Evaluation completed.")

    logger.info("Customer churn pipeline executed successfully")

except Exception as e:
    logger.error(f"Project failed: {e}", exc_info=True)
