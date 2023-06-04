import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("../data/FireDataset.csv")


columns_to_standardize = df.columns

# Perform standardization using StandardScaler
scaler = StandardScaler()
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])




fig, ax = plt.subplots(figsize=(12, 8))


boxplot = ax.boxplot(df.values, patch_artist=True, vert=False)


num_boxes = len(df.columns)
colors = [plt.cm.Set3(random.random()) for _ in range(num_boxes)]


for box, color in zip(boxplot['boxes'], colors):
    box.set_facecolor(color)


ax.set_yticklabels(df.columns)

plt.title("Boxplot of of Standardized Columns (Using standardscaler)")
plt.xlabel("Value")
plt.ylabel("Columns")
plt.tight_layout()
plt.show()
