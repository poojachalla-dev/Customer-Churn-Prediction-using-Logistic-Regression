from libraries import pd, np
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_name = os.path.join(BASE_DIR, "Data", "Telco_Customer_Churn_Dataset.csv")

df = pd.read_csv(file_name)

print(df.head())
print(df.info())
print(df.describe())