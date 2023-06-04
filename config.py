import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="Pipeline Options",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_path", type=str,
                        default="./dataset/FireDataset.csv")

    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--eval", type=str, default='kfold')  # holdout, kfold,
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument("--visual", action='store_true', default=False)

    # dt(decision tree), lr(logistic regression), knn(k-nearest neighbors), rf(random forest), ab(AdaBoost), gb(gradient boosting)
    
    parser.add_argument("--model", type=str, default='ab')
    # K-fold, split number
    parser.add_argument("--n_split", type=int, default=10)
    parser.add_argument("--target_number", type=int, default=3)
    return parser
