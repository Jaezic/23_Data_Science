from numpy import array
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from dataset.preprocess.preprocessing import preprocessing


class Dataset():
    """
        Dataset class, Base class for dataset
        Attributes:
            x: features
            y: target
    """
    def __init__(self, x, y):
        """
            Init function, if x and y have different length, raise ValueError
            Args:
                x: features
                y: target
        """
        self.x = x
        self.y = y
        if len(self.x) != len(self.y):
            raise ValueError('x and y must have the same length')

    def len(self):
        return len(self.x)

    def get(self):
        return self.x, self.y
    
    def PCA_pipeline(self, args, train_dataset, test_dataset):
        """
        Pipeline of PCA, and standardization
            Args:
                args: arguments from argument_parser()
                train_dataset: training dataset
                test_dataset: testing dataset
            
            return
                train_dataset: training dataset after PCA and standardization (Shared memory)
        """
        original_dim = train_dataset.x.shape[1]
        
        # Standardization
        scaler = StandardScaler()
        scaler.fit(train_dataset.x)
        train_dataset.x = scaler.transform(train_dataset.x)
        if test_dataset != None:
            test_dataset.x = scaler.transform(test_dataset.x)
        
        # PCA
        pca = PCA(n_components=args.n_components)
        pca.fit(train_dataset.x)
        train_dataset.x = pca.transform(train_dataset.x)
        if test_dataset != None:
            test_dataset.x = pca.transform(test_dataset.x)
        pca_dim = train_dataset.x.shape[1]
        print(f'<< PCA: {original_dim} -> {pca_dim} >>')

class FireDataset(Dataset):
    """
        FireDataset class, Dataset class for fire dataset
        Attributes:
            x: features
            y: target
            x_name: name of features
            y_name: name of target
            x_train: features of training dataset
            x_test: features of testing dataset
            y_train: target of training dataset
            y_test: target of testing dataset
    """
    def __init__(self, args):
        """
            Init function, read csv file, preprocessing, split dataset into training and testing dataset
            Args:
                args: arguments from argument_parser()
                
            Return:
                None
            """
        self.args = args
        df = pd.read_csv(args.data_path, na_filter=True,
                         keep_default_na=False, na_values=[''])
        df = preprocessing(args, df)
        df.to_csv('./dataset/preprocessed.csv', index=False)
        y = df['scale_damage'].values
        x = df.drop(['scale_damage'], axis=1).values
        super().__init__(x, y)

        self.x_name = df.drop(['scale_damage'], axis=1).columns.to_list()
        self.y_name = ['scale_damage']

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=args.seed, stratify=self.y, shuffle=True)
        
        
    def get_all(self):
        """
            Get all dataset
            Return:
                Dataset: all dataset
        """
        return Dataset(self.x, self.y)

    def get_train(self):
        """
            Get training dataset
            Return:
                Dataset: training dataset
        """
        return Dataset(self.x_train, self.y_train)

    def get_test(self):
        """
            Get testing dataset
            Return:
                Dataset: testing dataset
        """
        return Dataset(self.x_test, self.y_test)

    def get_kfold(self):
        """
            Get kfold dataset
            Return:
                kfold: kfold dataset, index of training and testing dataset
        """
        kfold = KFold(n_splits=self.args.n_split, shuffle=True,
                      random_state=self.args.seed)
        return kfold.split(self.x, self.y)
    

    def get_stratified_kfold(self):
        """
            Get stratified kfold dataset
            Return:
                kfold: stratified kfold dataset, index of training and testing dataset
        """
        kfold = StratifiedKFold(n_splits=self.args.n_split, shuffle=True,
                      random_state=self.args.seed)
        return kfold.split(self.x, self.y)


    