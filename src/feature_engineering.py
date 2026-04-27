def create_features(df):

    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["IsNewCustomer"] = (df["tenure"] < 12).astype(int)
    df["HighValueCustomer"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

    print(f"AvgMonthlySpend:", df["AvgMonthlySpend"])
    print(f"IsNewCustomer:", df["IsNewCustomer"])
    print(f"HighValueCustomer:", df["HighValueCustomer"])

    return df