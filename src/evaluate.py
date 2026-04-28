import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from logger import setup_logger

logger = setup_logger()


# =========================================================
# 1. DEFAULT EVALUATION (Threshold = 0.5)
# =========================================================
def evaluate_model(model, X_test, y_test):
    try:
        logger.info("Starting model evaluation (default threshold = 0.5)...")

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        # Logging
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"ROC AUC: {roc:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Classification Report:\n{cr}")

        # Print
        print("\n=== DEFAULT THRESHOLD (0.5) RESULTS ===")
        print("Accuracy:", acc)
        print("ROC AUC:", roc)
        print(cm)
        print(cr)

        logger.info("Default evaluation completed successfully.")

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}", exc_info=True)
        raise


# =========================================================
# 2. THRESHOLD OPTIMIZATION
# =========================================================
def find_best_threshold(model, X_test, y_test):
    try:
        logger.info("Starting threshold optimization...")

        y_prob = model.predict_proba(X_test)[:, 1]

        thresholds = np.arange(0.1, 0.9, 0.05)

        best_threshold = 0.5
        best_f1 = 0

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_test, y_pred)

            logger.info(f"Threshold: {t:.2f} | F1 Score: {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        logger.info(f"Best Threshold Found: {best_threshold}")
        logger.info(f"Best F1 Score: {best_f1:.4f}")

        return best_threshold

    except Exception as e:
        logger.error(f"Error during threshold optimization: {e}", exc_info=True)
        raise


# =========================================================
# 3. EVALUATION WITH OPTIMIZED THRESHOLD
# =========================================================
def evaluate_with_threshold(model, X_test, y_test, threshold):
    try:
        logger.info(f"Evaluating model with optimized threshold: {threshold}")

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        # Logging
        logger.info(f"[OPT] Accuracy: {acc:.4f}")
        logger.info(f"[OPT] ROC AUC: {roc:.4f}")
        logger.info(f"[OPT] Confusion Matrix:\n{cm}")
        logger.info(f"[OPT] Classification Report:\n{cr}")

        # Print
        print("\n=== OPTIMIZED THRESHOLD RESULTS ===")
        print("Accuracy:", acc)
        print("ROC AUC:", roc)
        print(cm)
        print(cr)

        logger.info("Optimized evaluation completed successfully.")

    except Exception as e:
        logger.error(f"Error during optimized evaluation: {e}", exc_info=True)
        raise
