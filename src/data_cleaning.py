from libraries import pd

def cleaned_data(df):

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)

    # This line converts the target column Churn from text labels into numeric values:
    df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

    print(df["Churn"].value_counts())

    return df