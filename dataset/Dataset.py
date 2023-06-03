from numpy import array
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.preprocess.preprocessing import preprocessing


class Dataset():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        if len(self.x) != len(self.y):
            raise ValueError('x and y must have the same length')

    def len(self):
        return len(self.x)

    def get(self):
        return self.x, self.y


class FireDataset(Dataset):
    def __init__(self, args):
        df = pd.read_csv(args.data_path, na_filter=True,
                         keep_default_na=False, na_values=[''])
        preprocessing(df)

        y = df['scale_damage'].values
        x = df.drop(['scale_damage'], axis=1).values
        super().__init__(x, y)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2)
        

    def get_train(self):
        return Dataset(self.x_train, self.y_train)
    
    def get_test(self):
        return Dataset(self.x_test, self.y_test)
