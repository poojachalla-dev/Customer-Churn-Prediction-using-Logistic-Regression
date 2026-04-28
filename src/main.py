from data_cleaning import cleaned_data
from feature_engineering import create_features
from reading_data import load_data
from eda import eda
from preprocess import build_preprocessor
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

# Preprocessing
preprocessor = build_preprocessor(X)
print(f"X.shape:", X.shape)
