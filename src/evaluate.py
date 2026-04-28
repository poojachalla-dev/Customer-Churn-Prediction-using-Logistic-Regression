from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from logger import setup_logger

logger = setup_logger()


def evaluate_model(model, X_test, y_test):
    try:
        logger.info("Starting model evaluation...")

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        # Logging results
        logger.info(f"Accuracy Score: {acc:.4f}")
        logger.info(f"ROC AUC Score: {roc:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Classification Report:\n{cr}")

        # Optional print output
        print("Accuracy:", acc)
        print("ROC AUC:", roc)
        print(cm)
        print(cr)

        logger.info("Model evaluation completed successfully.")

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}", exc_info=True)
