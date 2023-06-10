from collections import OrderedDict
import os
import pprint

from config import argument_parser
from tools.utils import ReDirectSTD, time_str
import pandas as pd
from main import main

if __name__ == '__main__':
    # Argument parsing
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
        
    # Pretty print the run args
    pprint.pprint(OrderedDict(args.__dict__))
    print('-' * 60) 
    
    # All Evaluation of parameters
    args.eval = 'kfold_stratified'
    df = pd.DataFrame(columns=['model', 'pca', 'standard','accuracy', 'precision', 'recall', 'f1'])

    tune_lists = [([None], False), (['grid',None], True)]
    
    # if you want to tune hyperparameters, set tune_config = tune_lists[1]
    # but, not want to tune hyperparameters, set tune_config = tune_lists[0]
    tune_config = tune_lists[0]

    for model in ['dt', 'knn', 'rf', 'ab', 'gb', 'kmeans', 'bag', 'voting']:
        for pca in [False, True]:
            for standard in [False, True]:
                for tune in tune_config[0]:
                    args.model = model
                    args.pca = pca
                    args.standard = standard
                    args.tune = tune
                    if args.pca == True and args.standard == False:
                        continue
                    if args.tune == 'grid':
                        args.param_load = False
                    else:
                        args.param_load = tune_config[1]

                    print('Model: {}, PCA: {}, Standard: {}, SMOTE: {}, Tune: {}, Param_load: {}'.format(args.model, args.pca, args.standard, args.smote, args.tune, args.param_load))
                    try:
                        acc, pre, rec, f1 = main(args)
                    except Exception as e:
                        print(e)
                        continue
                    # save result
                    if args.tune == None:
                        df_row = pd.DataFrame({'model': [args.model], 'pca': [args.pca], 'standard': [args.standard], 'accuracy': [acc], 'precision': [pre], 'recall': [rec], 'f1': [f1]})
                        df = pd.concat([df, df_row], axis=0)
                        print(df)
    # save result
    df.to_csv('result.csv', index=False)