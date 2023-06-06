from collections import OrderedDict
import os
import pprint

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from config import argument_parser
from dataset.Dataset import Dataset, FireDataset
from models.model import build_model
from tools.evaluate import evaluate
from tools.smote import smote
from tools.tune import tune_pipeline
from tools.utils import ReDirectSTD, set_seed, time_str
from tools.visualization import visual
import pandas as pd
import main

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    
    # Logging setup
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    if args.redirector:
        print('ReDirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)
    pprint.pprint(OrderedDict(args.__dict__))
    print('-' * 60) 
    
    df = pd.DataFrame(columns=['model', 'pca', 'standard', 'tune','accuracy', 'precision', 'recall', 'f1'])
    
    for model in ['dt', 'knn', 'rf', 'ab', 'gb', 'kmeans', 'bag', 'voting']:
        for pca in [False, True]:
            for standard in [False, True]:
                for tune in ['grid', None]:
                    args.model = model
                    args.pca = pca
                    args.standard = standard
                    args.tune = tune
                    if args.pca == True and args.standard == False:
                        continue
                    if args.tune == 'grid':
                        args.param_load = False
                    else:
                        args.param_load = True
                    print('Model: {}, PCA: {}, Standard: {}, SMOTE: {}, Tune: {}, Param_load: {}'.format(args.model, args.pca, args.standard, args.smote, args.tune, args.param_load))
                    try:
                        acc, pre, rec, f1 = main(args)
                    except:
                        print('Error')
                        continue
                    if args.tune == None:
                        df_row = pd.DataFrame({'model': [args.model], 'pca': [args.pca], 'standard': [args.standard], 'accuracy': [acc], 'precision': [pre], 'recall': [rec], 'f1': [f1]})
                        df = pd.concat([df, df_row], axis=0)
                        print(df)
    df.to_csv('result.csv', index=False)