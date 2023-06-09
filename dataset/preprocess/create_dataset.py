import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions.distance import calculate_distance

from functions.scale_damage import scale_damage


def main(args):
    # Read CSV
    FS_df = pd.read_csv(args.FireStatistic_root)
    FSP_df = pd.read_csv(args.FireStationPos_root, encoding='cp949')
    MH_df = pd.read_excel(args.MountainHeight_root)

    # Convert Address to Lat/Long
    # But, This code is very long time to run, so I comment out.
    # FS_address = 지번to도로명주소(FS_df)
    # FS_latlong = 도로명주소to위도경도(FS_address)

    # Calculate FireStation Number Based on Distance
    
    # Read CSV
    FSP_ll_df = pd.read_csv(args.FireStation_latlong_root, encoding='cp949', )
    FSP_ll_df.rename(columns={'위도': 'latitude'}, inplace=True)
    FSP_ll_df.rename(columns={'경도': 'longitude'}, inplace=True)

    # Read CSV
    FS_ll_df = pd.read_csv(args.FireStatistic_latlong_root, na_filter=True,
                           keep_default_na=False, na_values=['', ' ', '0'])
    FS_ll_df.rename(columns={'위도': 'latitude'}, inplace=True)
    FS_ll_df.rename(columns={'경도': 'longitude'}, inplace=True)

    # Calculate FireStation Number Based on Distance
    distance = calculate_distance(FS_ll_df, FSP_ll_df)
    is_all_nan = np.all(np.isnan(distance), axis=1)

    within_5km = np.sum(distance <= 5, axis=1).astype(np.float64)
    within_10km = np.sum(distance <= 10, axis=1).astype(np.float64)
    within_30km = np.sum(distance <= 30, axis=1).astype(np.float64)

    # Calculate FireFacility Number Based on Distance
    FF_ll_df = pd.read_csv(args.FireFacility_latlong_root, encoding='cp949')
    FF_ll_df.rename(columns={'위도': 'latitude'}, inplace=True)
    FF_ll_df.rename(columns={'경도': 'longitude'}, inplace=True)
    FF_ll_df = FF_ll_df[(FF_ll_df['시설유형코드'] == 3) | (FF_ll_df['시설유형코드'] == 4)]
    distance = calculate_distance(FS_ll_df, FF_ll_df)

    within_5km_fact = np.sum(distance <= 5, axis=1)
    within_10km_fact = np.sum(distance <= 10, axis=1)
    within_30km_fact = np.sum(distance <= 30, axis=1)

    # Calculate Mountain Height
    MH_ll_df = pd.read_csv(args.MountainHeight_latlong_root)
    MH_ll_df.rename(columns={'위도(N)': 'latitude'}, inplace=True)
    MH_ll_df.rename(columns={'경도(E)': 'longitude'}, inplace=True)
    distance = calculate_distance(FS_ll_df, MH_ll_df)
    
    # find nearest mountain height(MH_ll_df['높이(m)']) based on distance variable
    minindex = distance.argmin(axis=1)
    heights = MH_ll_df['높이(m)'][minindex].values

    # NaN Handling
    heights[is_all_nan] = np.nan

    within_5km[is_all_nan] = np.nan
    within_10km[is_all_nan] = np.nan
    within_30km[is_all_nan] = np.nan
    FS_df['exintgtm'].replace(' ', np.nan, inplace=True)

    # Feature Selection
    columns_to_drop = ['Unnamed: 0', 'extingdt', 
                       'ocurdt', 'ocuremd', 'ocurgm', 'ocurjibun', 'ocurri', 'ocursgg', 'ocuryoil', 'ownersec']
    FS_df = FS_df.drop(columns_to_drop, axis=1)

    # Compose FireStation DataFrame
    # (Fire Statistic + Fire Station & Facility's Number Based on Distance + Mountain Height + Scale Damage)
    FS_df['within_5km'] = within_5km
    FS_df['within_10km'] = within_10km
    FS_df['within_30km'] = within_30km
    FS_df['within_5km_fact'] = within_5km_fact
    FS_df['within_10km_fact'] = within_10km_fact
    FS_df['within_30km_fact'] = within_30km_fact
    FS_df['height'] = heights

    # Drop NaNs
    FS_df = FS_df.dropna(axis=0, how='any')

    # Calculate Scale Damage
    damge_df = scale_damage(FS_df)

    FS_df['scale_damage'] = damge_df['scale_damage']
    column = FS_df.columns.tolist()
    column.pop(column.index('scale_damage'))
    column.append('scale_damage')
    FS_df = FS_df[column]


    # Drop Unnecessary Columns
    FS_df = FS_df.drop(['dmgarea', 'dmgmoney', 'exintgtm'], axis=1)

    FS_df.to_csv('./dataset/FireDataset.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing Options",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--FireStatistic_root", type=str,
                        default="./dataset/Fire/FireStatistic.csv")
    parser.add_argument("--FireStationPos_root", type=str,
                        default="./dataset/Fire/FireStationPos.csv")
    parser.add_argument("--MountainHeight_root", type=str,
                        default="./dataset/Fire/MountainHeight.xlsx")
    parser.add_argument("--FireStation_latlong_root", type=str,
                        default="./dataset/Fire/FireStation_latlong.csv")
    parser.add_argument("--FireStatistic_latlong_root", type=str,
                        default="./dataset/Fire/FireStatistic_latlong.csv")
    parser.add_argument("--FireFacility_latlong_root", type=str,
                        default="./dataset/Fire/FireFacility_latlong.csv")
    parser.add_argument("--MountainHeight_latlong_root", type=str,
                        default="./dataset/Fire/MountainHeight_latlong.csv")

    args = parser.parse_args()
    main(args)
