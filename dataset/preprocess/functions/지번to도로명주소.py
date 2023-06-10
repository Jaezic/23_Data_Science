import requests
import xml.etree.ElementTree as elemTree
import pandas as pd
import numpy as np
from tqdm import tqdm


def api(text=''):
    """
        Find the address of the text
        Args:
            text: address
        Returns:
            tree: xml.etree.ElementTree
    """
    # API Request URL
    url = 'https://business.juso.go.kr/addrlink/addrLinkApi.do'
    params = {'confmKey': "devU01TX0FVVEgyMDIzMDUyMTEwMzExMTExMzc5MTg=",
              'currentPage': '1', 'countPerPage': '1', 'keyword': text}

    # API Request
    response = requests.get(url, params=params)

    # XML Parsing
    tree = elemTree.fromstring(response.content)
    return tree


def 지번to도로명주소(path_csv):  # '../forest_fire.csv'
    """
        Find the address of the text in the csv file
        Args:
            path_csv: path of the csv file
        Returns:
            find_juso: DataFrame
    """
    # Read csv file
    data = pd.read_csv(path_csv)
    backup_data = data.copy()
    data = data.loc[:, ['ocurdo', 'ocursgg', 'ocurjibun']]
    
    # concat address
    juso = data.apply(lambda x: x['ocurdo'] + ' ' +
                      x['ocursgg'] + ' ' + x['ocurjibun'], axis=1)

    find_juso = pd.DataFrame(columns=['juso'])
    
    # Find address
    for a in tqdm(juso):
        try:
            tree = api(a)
        except:
            print('except')
            find_juso = pd.concat([find_juso, pd.DataFrame(
                {'juso': [pd.NA]})], ignore_index=True)
            continue
        if tree[0][0].text == '0': # if not found address
            find_juso = pd.concat([find_juso, pd.DataFrame(
                {'juso': [pd.NA]})], ignore_index=True)
        else: # if found address
            find_juso = pd.concat([find_juso, pd.DataFrame(
                {'juso': [tree[1][0].text]})], ignore_index=True)

    # Null Check, and ReSearch
    nulls = backup_data.loc[find_juso['juso'].isna(
    ), ['ocurdo', 'ocuremd', 'ocurjibun', 'ocurgm', 'ocurri', 'ocursgg']]
    
    # new address, concat address
    nulls_juso = nulls[find_juso['juso'].isna()].apply(
        lambda x: x['ocurdo'] + ' ' + x['ocuremd'] + ' ' + x['ocurjibun'], axis=1)

    find_juso2 = pd.DataFrame(columns=['juso'])
    # Find address in nulls
    for a in tqdm(nulls_juso):
        try:
            tree = api(a)
        except:
            print('except')
            find_juso2 = pd.concat(
                [find_juso2, pd.DataFrame({'juso': [pd.NA]})], ignore_index=True)
            continue
        if tree[0][0].text == '0': # if not found address
            find_juso2 = pd.concat(
                [find_juso2, pd.DataFrame({'juso': [pd.NA]})], ignore_index=True)
        else: # if found address
            find_juso2 = pd.concat([find_juso2, pd.DataFrame(
                {'juso': [tree[1][0].text]})], ignore_index=True)
    
    find_juso2.set_index(nulls_juso.index, inplace=True) # set index
    find_juso.loc[find_juso['juso'].isna(), 'juso'] = find_juso2['juso'] # isna -> fillna
    find_juso.to_csv('./find_juso.csv', encoding='cp949')
    return find_juso