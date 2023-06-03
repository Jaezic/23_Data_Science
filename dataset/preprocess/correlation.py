import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv('./data/FireDataset.csv')
X=data.iloc[:,0:17]
y= data.iloc[:,-1]

corrmat = data.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(17,17))
g=sns.heatmap(data[top_corr_features].corr(),annot=True, cmap="RdYlGn")
plt.show()