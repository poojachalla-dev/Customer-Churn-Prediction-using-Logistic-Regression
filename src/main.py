from data_cleaning import cleaned_data
from feature_engineering import create_features
from reading_data import load_data
from eda import eda
from preprocess import build_preprocessor
from train import split_data
import os

# Load data
df = load_data()

# Data Cleaning
df = cleaned_data(df)

# Feature engineering
df = create_features(df)

# Save the data back to file
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(BASE_DIR, "Data", "Cleaned_Data.csv")

df.to_csv(file_path, index=False)

# EDA
df = eda(df)

# Split features
X = df.drop(["customerID", "Churn"], axis=1)
y = df["Churn"]

# Split Train and Test Set
X_train, X_test, y_train, y_test = split_data(X, y)

# Preprocessing
preprocessor = build_preprocessor(X_train)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"(X_train_processed:", X_train_processed.shape)
print(f"X_test_processed:", X_test_processed.shape)
