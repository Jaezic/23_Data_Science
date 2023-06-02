import argparse
import pandas as pd

from functions.scale_damage import scale_damage
from functions.도로명주소to위도경도 import 도로명주소to위도경도
from functions.지번to도로명주소 import 지번to도로명주소


def main(args):
    # Read CSV
    FS_df = pd.read_csv(args.FireStatistic_root)
    FFP_df = pd.read_csv(args.FireFightingPos_root, encoding='cp949')
    MH_df = pd.read_excel(args.MountainHeight_root)

    # Convert Address to Lat/Long
    # FS_address = 지번to도로명주소(FS_df)
    # FS_latlong = 도로명주소to위도경도(FS_address)

    # Calculate Scale Damage
    damge_df = scale_damage(FS_df)
    print(damge_df)

    # Compose FireStation DataFrame 
    # (Fire Statistic + Fire Station & Facility's Number Based on Distance + Mountain Height + Scale Damage)

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing Options",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--FireStatistic_root", type=str,
                        default="./dataset/Fire/FireStatistic.csv")
    parser.add_argument("--FireFightingPos_root", type=str,
                        default="./dataset/Fire/FireFightingPos.csv")
    parser.add_argument("--MountainHeight_root", type=str,
                        default="./dataset/Fire/MountainHeight.xlsx")

    args = parser.parse_args()
    main(args)
