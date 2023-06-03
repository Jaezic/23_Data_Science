import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd


data=pd.read_csv('./dataset/FireDataset.csv')
df=pd.DataFrame(data)
covMatrix = pd.DataFrame.cov(df)
print (covMatrix)

sn.heatmap(covMatrix, annot = True, fmt='g')
plt.show()

