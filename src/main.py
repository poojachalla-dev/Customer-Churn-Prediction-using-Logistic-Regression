from data_cleaning import cleaned_data
from feature_engineering import create_features
from reading_data import df

# Data Cleaning
df = cleaned_data(df)

# Feature engineering
df = create_features(df)