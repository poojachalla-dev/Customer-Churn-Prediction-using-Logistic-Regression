from libraries import plt, sns
from reading_data import df

plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Count")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot((df["tenure"]), bins=30, kde=True)
plt.title("Tenure Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges by Churn")
plt.show()

plt.figure(figsize=(8,4))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Contract Type vs Churn")
plt.show()
