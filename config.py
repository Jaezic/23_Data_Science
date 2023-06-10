import argparse


def argument_parser():
    """
    Argument parser
        Summary: define the arguments, and return the parser
        
        Returns:
            parser: parser
    """
    """
    Source : https://docs.python.org/ko/3/library/argparse.html
    What is argparse?
        The argparse module makes it easy to write user-friendly command-line interfaces.
        The program defines what arguments it requires, and argparse will figure out how to parse those out of sys.argv.
        The argparse module also automatically generates help and usage messages and issues errors when users give the program invalid arguments.
    
    What is ArgumentParser?
        The ArgumentParser object will hold all the information necessary to parse the command line into Python data types.
        The argparse module includes tools for building command line argument and option processors.
    
    What is add_argument()?
        The add_argument() method is used to specify which command-line options the program is willing to accept.
        This method is called once for each command-line option that the program is willing to accept.
    """
    parser = argparse.ArgumentParser(description="Pipeline Options",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_path", type=str,
                        default="./dataset/FireDataset.csv", help="Path to the dataset")
    parser.add_argument("--param_path", type=str,
                        default="./models/config", help="Path to the hyperparameters")
    parser.add_argument("--seed", type=int, default=64, help="Random seed")
    parser.add_argument("--redirector", action='store_false', default=True, help="Redirect stdout")
    parser.add_argument("--visual", action='store_true', default=False, help="Visualize the data")

    # 'dt'(decision tree), 'lr'(logistic regression), 'knn'(k-nearest neighbors), 'rf'(random forest),
    # 'ab'(AdaBoost), 'gb'(gradient boosting), 'voting' (voting classifier), 'bag' (bagging classifier)
    # 'kmeans' (k-means clustering)
    parser.add_argument("--model", type=str, default='rf', help="For Training model")
    parser.add_argument("--param_load", action='store_true', default=False, help="Load the hyperparameters")
    # Voting model list
    parser.add_argument("--voting_list", type=str, nargs='+',
                        default=['dt', 'knn', 'rf', 'ab', 'gb'], help='Example: --voting_list dt lr knn rf ab gb')

    # Hyperparameters tuning
    # 'grid'(grid search), 'random'(random search), None
    parser.add_argument("--tune", type=str, default=None, help="Tune hyperparameters mode")
    parser.add_argument("--n_iter", type=int, default=10, help="Number of iterations for random search")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross validation for grid search")

    # PCA
    parser.add_argument("--pca", action='store_true', default=False, help="PCA mode")
    parser.add_argument("--n_components", type=float, default=0.95, help="Number of components for PCA")

    # SMOTE
    parser.add_argument("--smote", action='store_true', default=False, help="SMOTE mode")

    # Standardization
    parser.add_argument("--standard", action='store_true', default=False, help="Standardization mode")
    
    # K-fold, split number
    parser.add_argument("--eval", type=str, default='holdout', help='Evaluation Method')  # holdout, kfold, kfold_stratified
    parser.add_argument("--n_split", type=int, default=10, help='Number of split for k-fold cross validation')
    parser.add_argument("--num_class", type=int, default=3, help='Number of classes for target')

    return parser
