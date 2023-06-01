from numpy import array
import pandas as pd


class FireDataset():
    def __init__(self, args):
        df = pd.read_csv(args.data_path)
        self.x = df[['SAT']].to_numpy()
        self.y = df[['GPA']].to_numpy()

    def len(self):
        return len(self.x)

    
    def getSet(self):
        return self.x, self.y
