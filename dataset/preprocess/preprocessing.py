import argparse
import pandas as pd
from functions.distance import calculate_distance

from functions.scale_damage import scale_damage
from functions.도로명주소to위도경도 import 도로명주소to위도경도
from functions.지번to도로명주소 import 지번to도로명주소


def main(args):
    # Read CSV
    FS_df = pd.read_csv(args.FireStatistic_root)
    FSP_df = pd.read_csv(args.FireStationPos_root, encoding='cp949')
    MH_df = pd.read_excel(args.MountainHeight_root)

    # Convert Address to Lat/Long
    # FS_address = 지번to도로명주소(FS_df)
    # FS_latlong = 도로명주소to위도경도(FS_address)

    # Calculate Scale Damage
    # damge_df = scale_damage(FS_df)
    # print(damge_df)

    # Calculate Distance
    FSP_ll_df = pd.read_csv(args.FireStation_latlong_root, encoding='cp949')
    FSP_ll_df.rename(columns={'위도': 'latitude'}, inplace=True)
    FSP_ll_df.rename(columns={'경도': 'longitude'}, inplace=True)

    FS_ll_df = pd.read_csv(args.FireStatistic_latlong_root)
    FS_ll_df.rename(columns={'위도': 'latitude'}, inplace=True)
    FS_ll_df.rename(columns={'경도': 'longitude'}, inplace=True)
    print(calculate_distance(FS_ll_df, FSP_ll_df).shape)
    # Compose FireStation DataFrame
    # (Fire Statistic + Fire Station & Facility's Number Based on Distance + Mountain Height + Scale Damage)

    # FS_df['scale_damage'] = damge_df['scale_damage']

    pass


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

    args = parser.parse_args()
    main(args)
