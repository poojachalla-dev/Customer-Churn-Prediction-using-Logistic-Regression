import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def build_preprocessor(X):
    try:
        logging.info("Starting preprocessor creation process")

        logging.info("Identifying numerical columns")
        num_cols = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        logging.info(f"Numerical columns found: {num_cols}")

        logging.info("Identifying categorical columns")
        cat_cols = X.select_dtypes(
            include=["object"]
        ).columns.tolist()

        logging.info(f"Categorical columns found: {cat_cols}")

        logging.info("Creating ColumnTransformer")

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])

        logging.info("Preprocessor created successfully")
        logging.info(
            "Transformers configured: "
            "StandardScaler for numerical columns, "
            "OneHotEncoder for categorical columns"
        )

        return preprocessor

    except Exception as e:
        logging.exception("Error occurred while building preprocessor")
        raise
