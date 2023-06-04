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

def main(args):
    set_seed(args.seed)  # Set random seed

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

    # Dataset setup
    dataset = FireDataset(args)
    print(f'Dataset size: {dataset.len()}')
    print(f'Features name : {dataset.x_name}')
    print(f'Target name : {dataset.y_name}')
    print('-' * 60)

    # Model setup
    model = build_model(args)

    # Train and evaluate
    if args.tune != None:
        dataset = dataset.get_all()
        tune_pipeline(args, model, dataset)

    elif args.eval == 'holdout':
        train_dataset = dataset.get_train()
        test_dataset = dataset.get_test()

        metrics = pipeline(args, model, train_dataset, test_dataset)
    elif args.eval == 'kfold':
        kfold = dataset.get_kfold()
        metrics_list = []
        for i, (train_index, test_index) in enumerate(kfold):
            print(f'Fold {i}')
            train_dataset = Dataset(
                dataset.x[train_index], dataset.y[train_index])
            test_dataset = Dataset(
                dataset.x[test_index], dataset.y[test_index])

            metrics = pipeline(args, model, train_dataset, test_dataset)
            metrics_list.append(metrics)

        acc = sum([m.accuracy for m in metrics_list]) / len(metrics_list)
        pre = sum([m.precision for m in metrics_list]) / len(metrics_list)
        rec = sum([m.recall for m in metrics_list]) / len(metrics_list)
        f1 = sum([m.f1 for m in metrics_list]) / len(metrics_list)
        
        print('Average metrics:')
        print(' Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(
            acc,
            pre,
            rec,
            f1
        ))
        return acc, pre, rec, f1

    
def pipeline(args, model, train_dataset, test_dataset):
    if args.pca and args.standard == False:
        raise ValueError('PCA must be used with standardization')
        
    if args.pca:
        train_dataset.PCA_pipeline(args, train_dataset, test_dataset)

    if args.smote:
        train_dataset.x, train_dataset.y = smote(args,train_dataset.x, train_dataset.y)
    model.fit(train_dataset.x, train_dataset.y)

    y = model.predict(test_dataset.x)

    #visual(dataset, y)

    metrics = evaluate(args, model, test_dataset.y, y)

    return metrics


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    # df = pd.DataFrame(columns=['model', 'pca', 'standard', 'accuracy', 'precision', 'recall', 'f1'])
    # for model in ['dt', 'knn', 'rf', 'ab', 'gb', 'kmeans', 'bag', 'voting' ]:
    #     for pca in [True, False]:
    #         for standard in [True, False]:
    #             args.model = model
    #             args.pca = pca
    #             args.standard = standard
    #             if args.pca == True and args.standard == False:
    #                 continue
    #             print('Model: {}, PCA: {}, Standard: {}, SMOTE: {}, Tune: {}'.format(args.model, args.pca, args.standard, args.smote, args.tune))
    #             acc, pre, rec, f1 = main(args)
    #             df_row = pd.DataFrame({'model': [args.model], 'pca': [args.pca], 'standard': [args.standard], 'accuracy': [acc], 'precision': [pre], 'recall': [rec], 'f1': [f1]})
    #             df = pd.concat([df, df_row], axis=0)
    #             print(df)
    # df.to_csv('result.csv', index=False)
    
    df = pd.DataFrame(columns=['model', 'pca', 'standard', 'tune','accuracy', 'precision', 'recall', 'f1'])
    
    for model in ['dt', 'knn', 'rf', 'ab', 'gb', 'kmeans', 'bag', 'voting']:
        for pca in [True, False]:
            for standard in [True, False]:
                for tune in ['grid', None]:
                    args.model = model
                    args.pca = pca
                    args.standard = standard
                    args.tune = tune
                    if args.pca == True and args.standard == False:
                        continue
                    print('Model: {}, PCA: {}, Standard: {}, SMOTE: {}, Tune: {}'.format(args.model, args.pca, args.standard, args.smote, args.tune))
                    acc, pre, rec, f1 = main(args)
                    df_row = pd.DataFrame({'model': [args.model], 'pca': [args.pca], 'standard': [args.standard], 'tune': [args.tune], 'accuracy': [acc], 'precision': [pre], 'recall': [rec], 'f1': [f1]})
                    df = pd.concat([df, df_row], axis=0)
                    print(df)
    df.to_csv('result.csv', index=False)