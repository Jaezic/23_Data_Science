import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="Pipeline Options",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_path", type=str,
                        default="./dataset/FireDataset.csv")
    parser.add_argument("--param_path", type=str,
                        default="./models/config")
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument("--visual", action='store_true', default=False)

    # 'dt'(decision tree), 'lr'(logistic regression), 'knn'(k-nearest neighbors), 'rf'(random forest),
    # 'ab'(AdaBoost), 'gb'(gradient boosting), 'voting' (voting classifier), 'bag' (bagging classifier)
    # 'kmeans' (k-means clustering)
    parser.add_argument("--model", type=str, default='voting')
    parser.add_argument("--param_load", action='store_true', default=False)
    # Voting model list
    parser.add_argument("--voting_list", type=str, nargs='+',
                        default=['dt', 'knn', 'rf', 'ab', 'gb'], help='Example: --voting_list dt lr knn rf ab gb')

    # Hyperparameters tuning
    # 'grid'(grid search), 'random'(random search), None
    parser.add_argument("--tune", type=str, default=None)
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--cv", type=int, default=5)

    # PCA
    parser.add_argument("--pca", action='store_true', default=False)
    parser.add_argument("--n_components", type=float, default=0.95)

    # SMOTE
    parser.add_argument("--smote", action='store_true', default=False)

    # Standardization
    parser.add_argument("--standard", action='store_true', default=True)
    
    # K-fold, split number
    parser.add_argument("--eval", type=str, default='kfold')  # holdout, kfold,
    parser.add_argument("--n_split", type=int, default=10)
    parser.add_argument("--num_class", type=int, default=3)

    return parser
