import pandas as pd
import numpy as np


def scale_damage(df):
    # covert object type to float type if convertable, otherwise NaN
    df['exintgtm'] = pd.to_numeric(df['exintgtm'], errors='coerce')

    # Standardization
    df['dmgarea'] = z_score(df['dmgarea'])
    df['dmgmoney'] = z_score(df['dmgmoney'])
    df['exintgtm'] = z_score(df['exintgtm'])

    # Scaling
    area_scale = 1.5
    money_scale = 1
    time_scale = 1.5
    scaling(df,area_scale, money_scale, time_scale)

    df['scale_damage'] = df['dmgarea'] + df['dmgmoney'] + df['exintgtm']

    result = df[['scale_damage']]
    return result


def scaling(df, area_scale, money_scale, time_scale):
    df['dmgarea'] = df['dmgarea']*area_scale
    df['dmgmoney'] = df['dmgmoney']*money_scale
    df['exintgtm'] = df['exintgtm']/time_scale


def z_score(data):
    mean = np.mean(data)  # 평균 계산
    std = np.std(data)    # 표준편차 계산

    z_scores = (data - mean) / std  # Z 점수 계산

    return z_scores
