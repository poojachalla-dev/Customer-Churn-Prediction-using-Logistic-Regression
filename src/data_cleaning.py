from libraries import pd
from reading_data import df

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)

# This line converts the target column Churn from text labels into numeric values:
df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

print(f"Churn:", df["Churn"])