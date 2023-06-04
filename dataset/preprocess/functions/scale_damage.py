import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer


def scale_damage(df):
    # covert object type to float type if convertable, otherwise NaN
    df['exintgtm'] = pd.to_numeric(df['exintgtm'], errors='coerce')


    scaler = RobustScaler()
    df[['dmgarea', 'dmgmoney', 'exintgtm']] = scaler.fit_transform(
        df[['dmgarea', 'dmgmoney', 'exintgtm']])
    scaler = Normalizer()
    df[['dmgarea', 'dmgmoney', 'exintgtm']] = scaler.fit_transform(
        df[['dmgarea', 'dmgmoney', 'exintgtm']])
    
    # Scaling
    area_scale = 1
    money_scale = 1
    time_scale = 1
    scaling(df,area_scale, money_scale, time_scale)

    df['scale_damage'] = df['dmgarea'] + df['dmgmoney'] + df['exintgtm']
    result = df[['scale_damage']]
    return result


def scaling(df, area_scale, money_scale, time_scale):
    df['dmgarea'] = df['dmgarea']*area_scale
    df['dmgmoney'] = df['dmgmoney']*money_scale
    df['exintgtm'] = df['exintgtm']/time_scale

