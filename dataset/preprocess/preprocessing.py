import argparse

import pandas as pd


def main(args):
    df = pd.read_csv(args.FireStatistic_root, na_filter=True, keep_default_na=False, na_values=[''])
    print(df.info())
    print(df.head())
    print(df.describe())
    print(df.isna().sum())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing Options",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--FireStatistic_root", type=str,
                        default="./dataset/FireDataset.csv")

    args = parser.parse_args()
    main(args)
