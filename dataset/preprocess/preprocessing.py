import argparse

import pandas as pd
from sklearn.calibration import LabelEncoder


def main(args):
    df = pd.read_csv(args.FireStatistic_root, na_filter=True, keep_default_na=False, na_values=[''])

    le=LabelEncoder()
    df['diravg'] = le.fit_transform(df['diravg'])
    df['dirmax'] = le.fit_transform(df['dirmax'])
    df['ocurcause'] = le.fit_transform(df['ocurcause'])
    df.to_csv(args.FireStatistic_root, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing Options",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--FireStatistic_root", type=str,
                        default="./dataset/FireDataset.csv")

    args = parser.parse_args()
    main(args)
