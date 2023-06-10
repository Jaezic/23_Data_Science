from dataset.Dataset import Dataset, FireDataset
from models.model import build_model
from tools.evaluate import evaluate
from tools.smote import smote
from tools.tune import tune_pipeline 
from tools.utils import set_seed
import pandas as pd

def main(args):
    """
    Main function
        Summary:  
            Set random seed, setup dataset and model, train and evaluate
        
        Args:
            args: arguments from argument_parser()
        
        Returns:
            metrics: metrics of the model
    """
    set_seed(args.seed)  # Set random seed

    # Dataset setup
    dataset = FireDataset(args)
    print(f'Dataset size: {dataset.len()}')
    print(f'Features name : {dataset.x_name}')
    print(f'Target name : {dataset.y_name}')
    print('-' * 60)

    # Model setup
    model = build_model(args)
    
    if args.tune != None:  # Tune hyperparameters
        dataset = dataset.get_all()
        # Tune and evaluate
        tune_pipeline(args, model, dataset)

    elif args.eval == 'holdout': # Holdout
        train_dataset = dataset.get_train()
        test_dataset = dataset.get_test()

        # Train and evaluate
        metrics = pipeline(args, model, train_dataset, test_dataset)
    elif 'kfold' in args.eval: # K-fold cross validation 
        if args.eval == 'kfold_stratified':
            kfold = dataset.get_stratified_kfold()
        else:
            kfold = dataset.get_kfold()
        metrics_list = []
        # K-fold cross validation
        for i, (train_index, test_index) in enumerate(kfold):
            print(f'Fold {i}')
            train_dataset = Dataset(
                dataset.x[train_index], dataset.y[train_index])
            test_dataset = Dataset(
                dataset.x[test_index], dataset.y[test_index])

            # Train and evaluate
            metrics = pipeline(args, model, train_dataset, test_dataset)
            metrics_list.append(metrics)

        # Calculate average metrics
        acc = sum([m.accuracy for m in metrics_list]) / len(metrics_list)
        pre = sum([m.precision for m in metrics_list]) / len(metrics_list)
        rec = sum([m.recall for m in metrics_list]) / len(metrics_list)
        f1 = sum([m.f1 for m in metrics_list]) / len(metrics_list)
        
        # Print average metrics
        print('Average metrics:')
        print(' Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(
            acc,
            pre,
            rec,
            f1
        ))
        return acc, pre, rec, f1
    return 0, 0, 0, 0

    
def pipeline(args, model, train_dataset, test_dataset):
    """
    Pipeline of training and evaluating
        Summary:
            PCA, SMOTE, train and evaluate
        
        Args:
            args: arguments from argument_parser()
            model: model to train
            train_dataset: training dataset
            test_dataset: testing dataset
        
        Returns:
            metrics: metrics of the model
    """
    if args.pca and args.standard == False:
        raise ValueError('PCA must be used with standardization')
        
    if args.pca:
        train_dataset.PCA_pipeline(args, train_dataset, test_dataset)

    if args.smote:
        train_dataset.x, train_dataset.y = smote(args,train_dataset.x, train_dataset.y)
    model.fit(train_dataset.x, train_dataset.y)

    y = model.predict(test_dataset.x)

    metrics = evaluate(args, model, test_dataset.y, y)

    return metrics