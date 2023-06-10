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
    pprint.pprint(OrderedDict(args.__dict__))
    print('-' * 60)
    
    # Main
    acc, pre, rec, f1 = main(args)
    
    