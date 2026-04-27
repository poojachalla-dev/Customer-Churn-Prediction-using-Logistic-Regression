from libraries import pd
from reading_data import df, file_name

def cleaned_data(df):

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)

    # Save back to same file
    df.to_csv(file_name, index=False)

    # This line converts the target column Churn from text labels into numeric values:
    df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

    print(f"Churn:", df["Churn"])

    return(df)