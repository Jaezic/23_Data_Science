from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer


def preprocessing(args, df):
    """
    Preprocessing function, including reflect feature engineering results, label encoding, binning, standardization
        Args:
            args: arguments from argument_parser()
            df: dataframe, including features and target
        Returns:
            df: dataframe, including features and target after preprocessing
    """
    feature_Handling(df)

    # Label Encoding
    le = LabelEncoder()
    df['diravg'] = le.fit_transform(df['diravg'])
    df['ocurcause'] = le.fit_transform(df['ocurcause'])
    df['ocurdo'] = le.fit_transform(df['ocurdo'])


    # Drop features that are less correlated with target
    df.drop(['within_5km', 'within_10km', 'within_5km_fact',
            'within_10km_fact'], axis=1, inplace=True)

    # df = df.loc[:,['height', 'within_5km', 'within_10km','within_30km', 'windavg','tempavg','scale_damage']]
    # Binning (scale_damage)
    discretizer = KBinsDiscretizer(
        n_bins=args.num_class, encode='ordinal', strategy='uniform')
    df['scale_damage'] = discretizer.fit_transform(
        df['scale_damage'].values.reshape(-1, 1))

    # Standardization
    if args.standard:
        scaler = StandardScaler()
        scaler.fit(df.drop(['scale_damage'], axis=1))
        df[df.drop(['scale_damage'], axis=1).columns] = scaler.transform(
            df.drop(['scale_damage'], axis=1))

    return df


def feature_Handling(df):
    """
        reflect Feature Handling results to dataframe, including occurcause handling, feature drop, feature rename
        Args:
            df: dataframe, including features and target
        Returns:
            None
    """
    # Ocurcause Handling
    df['ocurcause'] = df['ocurcause'].apply(
        lambda x: x.replace('추정', '') if '추정' in x else x)
    df['ocurcause'] = df['ocurcause'].apply(
        lambda x: '담배' if '담뱃' in x or '담배' in x else x)
    df['ocurcause'] = df['ocurcause'].apply(
        lambda x: '입산자실화' if '입산자' in x or '실화' in x or '발화' in x or '행위' in x or '훈련' in x else x)
    df['ocurcause'] = df['ocurcause'].apply(lambda x: '소각' if '소각' in x else x)
    df['ocurcause'] = df['ocurcause'].apply(lambda x: '방화' if '방화' in x else x)
    df['ocurcause'] = df['ocurcause'].apply(lambda x: '미상' if '미상' in x else x)
    df['ocurcause'] = df['ocurcause'].apply(lambda x: '화재' if '화재' in x else x)
    df['ocurcause'] = df['ocurcause'].apply(
        lambda x: '불장난' if '장난' in x else x)
    cnt = df['ocurcause'].value_counts()
    df['ocurcause'] = df['ocurcause'].apply(
        lambda x: '기타' if cnt[x] < 37 else x)

    # Feature Drop, [humidmin, humidcurr, windmax, dirmax, riskmax, riskavg]
    df.drop(['humidmin', 'humidcurr', 'windmax', 'dirmax',
            'riskmax', 'riskavg'], axis=1, inplace=True)
    
    # Feature Rename
    df['ocurdo'] = df['ocurdo'].replace('출북', '충북')
    df['ocurdo'] = df['ocurdo'].replace('서부', '전북')
    
    # diravg lowercase
    df['diravg'] = df['diravg'].str.lower()
    # delete dirty data('정온') on diravg
    df['diravg'] = df['diravg'].str.replace(pat=r'정온', repl=r'', regex=True)
    # drop number on diravg
    df['diravg'] = df['diravg'].str.replace(pat=r'[0-9]', repl=r'', regex=True)
    # drop space on diravg
    df['diravg'] = df['diravg'].str.replace(pat=r' ', repl=r'', regex=True)

    # drop empty data on diravg
    df['diravg'] = df['diravg'][df['diravg'] != '']

