import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="Pipeline Options",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_path", type=str,
                        default="./dataset/FireDataset.csv")
    parser.add_argument("--param_path", type=str,
                        default="./models/config")
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--eval", type=str, default='kfold')  # holdout, kfold,
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument("--visual", action='store_true', default=False)

    # 'dt'(decision tree), 'lr'(logistic regression), 'knn'(k-nearest neighbors), 'rf'(random forest), 'ab'(AdaBoost), 'gb'(gradient boosting)
    parser.add_argument("--model", type=str, default='dt')

    # Hyperparameters tuning
    # 'grid'(grid search), 'random'(random search), None
    parser.add_argument("--tune", type=str, default=None)
    parser.add_argument("--cv", type=int, default=5)

    # PCA
    parser.add_argument("--pca", action='store_true', default=False)
    parser.add_argument("--n_components", type=float, default=0.95)

    # K-fold, split number
    parser.add_argument("--n_split", type=int, default=10)
    parser.add_argument("--target_number", type=int, default=3)

    return parser
