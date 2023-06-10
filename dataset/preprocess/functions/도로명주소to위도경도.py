import requests
import pandas as pd
from tqdm import tqdm


# df = pd.read_csv('./find_juso_complete_212_na.csv')
def 도로명주소to위도경도(df):
    """
        Find the latitude and longitude of the address in the csv file
        Args:
            path_csv: path of the csv file
        Returns:
            df: DataFrame
    """
    
    # API Request URL
    url_front = "http://api.vworld.kr/req/address?"
    url_params = "service=address&request=getcoord&version=2.0&crs=epsg:4326&refine=true&simple=false&format=json&type=road"
    url_address = "&address="
    url_key = "&key="

    address = ""
    auth_key = "027669BD-097C-38AA-AADD-FD28E07F776E"


    x_list = []
    y_list = []

    # 도로명주소 -> 위도, 경도
    for row in tqdm(df.itertuples()):
        address = row[1] # 도로명주소
        try:
            url = url_front + url_params + url_address + address + url_key + auth_key # create url

            result = requests.get(url) # api request
            json_data = result.json() # json parsing

            if json_data['response']['status'] == 'OK': # if found address
                x = json_data['response']['result']['point']['x'] # Longitude
                y = json_data['response']['result']['point']['y'] # Latitude
                
                x_list.append(x)
                y_list.append(y)
                
            else: # if not found address
                x_list.append(pd.NA) # NaN
                y_list.append(pd.NA) # NaN
        except:
            x_list.append(pd.NA) # NaN
            y_list.append(pd.NA) # NaN
            
    df['경도'] = pd.Series(data = x_list, index = df.index)
    df['위도'] = pd.Series(data = y_list, index = df.index)

    print(df)

    df.to_csv('./FireStation_xy.csv', encoding='cp949')
    return df



