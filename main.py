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
from tools.tune import tune
from tools.utils import ReDirectSTD, set_seed, time_str
from tools.visualization import visual


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
        tune(args, model, dataset)

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

        print('Average metrics:')
        print(' Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(
            sum([m.accuracy for m in metrics_list]) / len(metrics_list),
            sum([m.precision for m in metrics_list]) / len(metrics_list),
            sum([m.recall for m in metrics_list]) / len(metrics_list),
            sum([m.f1 for m in metrics_list]) / len(metrics_list)
        ))

    
def pipeline(args, model, train_dataset, test_dataset):
    if args.pca and args.standard == False:
        raise ValueError('PCA must be used with standardization')
        
    if args.pca:
        FireDataset.PCA_pipeline(args, train_dataset, test_dataset)

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
    main(args)
