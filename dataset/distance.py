from haversine import haversine


def calculate_distance(df1, df2):
    df2['distance'] = df2.apply(lambda row: haversine((df1['latitude'][0], df1['longitude'][0]),
                                                      (row['latitude'], row['longitude'])), axis=1)
    return df2['distance']
