import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("../data/FireDataset.csv")

# Select the columns to be standardized
columns_to_standardize = df.columns

# Perform standardization using StandardScaler
scaler = StandardScaler()
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])


plt.figure(figsize=(12, 8))
for column in columns_to_standardize:
    plt.scatter(df[column], df[column], label=column)

plt.title("Scatter Plot of Standardized Columns (Using standardscaler)")
plt.xlabel("Value")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()
