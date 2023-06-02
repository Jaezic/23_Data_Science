import argparse

import pandas as pd
from sklearn.calibration import LabelEncoder

def main(args):
    df = pd.read_csv(args.FireStatistic_root, na_filter=True,
                     keep_default_na=False, na_values=[''])
    
    df = feature_Engineering(df)

    # Label Encoding
    le = LabelEncoder()
    df['diravg'] = le.fit_transform(df['diravg'])
    df['dirmax'] = le.fit_transform(df['dirmax'])
    df['ocurcause'] = le.fit_transform(df['ocurcause'])
    df.to_csv(args.FireStatistic_root, index=False)

def feature_Engineering(df):
    df['ocurcause'] = df['ocurcause'].apply(lambda x: x.replace('추정','') if '추정' in x else x)
    df['ocurcause'] = df['ocurcause'].apply(lambda x: '담배' if '담뱃' in x or '담배' in x else x)
    df['ocurcause'] = df['ocurcause'].apply(lambda x: '입산자실화' if '입산자' in x or '실화' in x or '발화' in x or '행위' in x or '훈련' in x else x)
    df['ocurcause'] = df['ocurcause'].apply(lambda x: '소각' if '소각' in x else x)
    df['ocurcause'] = df['ocurcause'].apply(lambda x: '방화' if '방화' in x else x)
    df['ocurcause'] = df['ocurcause'].apply(lambda x: '미상' if '미상' in x else x)
    df['ocurcause'] = df['ocurcause'].apply(lambda x: '화재' if '화재' in x else x)
    df['ocurcause'] = df['ocurcause'].apply(lambda x: '불장난' if '장난' in x else x)
    cnt = df['ocurcause'].value_counts()
    df['ocurcause'] = df['ocurcause'].apply(lambda x: '기타' if cnt[x] < 37 else x)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing Options",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--FireStatistic_root", type=str,
                        default="./dataset/FireDataset.csv")

    args = parser.parse_args()
    main(args)
