import requests
import pandas as pd
from tqdm import tqdm


# df = pd.read_csv('./find_juso_complete_212_na.csv')
def 도로명주소to위도경도(df):
    url_front = "http://api.vworld.kr/req/address?"
    url_params = "service=address&request=getcoord&version=2.0&crs=epsg:4326&refine=true&simple=false&format=json&type=road"
    url_address = "&address="
    url_key = "&key="

    address = ""
    auth_key = "027669BD-097C-38AA-AADD-FD28E07F776E"



    x_list = []
    y_list = []

    for row in tqdm(df.itertuples()):
        address = row[1]
        try:
            url = url_front + url_params + url_address + address + url_key + auth_key

            result = requests.get(url)
            json_data = result.json()

            if json_data['response']['status'] == 'OK':
                x = json_data['response']['result']['point']['x']
                y = json_data['response']['result']['point']['y']
                
                x_list.append(x)
                y_list.append(y)
                
            else:
                x_list.append(0)
                y_list.append(0)
        except:
            x_list.append(0)
            y_list.append(0)
            
    df['경도'] = pd.Series(data = x_list, index = df.index)
    df['위도'] = pd.Series(data = y_list, index = df.index)

    print(df)

    df.to_csv('./FireStation_xy.csv', encoding='cp949')
    return df



