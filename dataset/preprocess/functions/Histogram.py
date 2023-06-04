import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("../data/FireDataset.csv")


columns_to_standardize = df.columns

# Perform standardization using StandardScaler
scaler = StandardScaler()
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])


plt.figure(figsize=(12, 8))
df.hist(alpha=0.7, bins=20, grid=False, edgecolor='black')

plt.title("Histogram of Standardized Columns (Using standardscaler)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
