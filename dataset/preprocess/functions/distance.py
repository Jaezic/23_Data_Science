import numpy as np


def haversine_numpy(lat1, lon1, lat2, lon2):
    R = 6371.0  # radius of the Earth in kilometers

    # Convert latitudes and longitudes to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Calculate differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Apply Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * \
        np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Calculate distance
    distance = R * c
    return distance


def calculate_distance(df1, df2):
    df1_lat = df1['latitude'].to_numpy()
    df1_lon = df1['longitude'].to_numpy()

    df2_lat = df2['latitude'].to_numpy()
    df2_lon = df2['longitude'].to_numpy()
    print(haversine_numpy(df1_lat[0], df1_lon[0], df2_lat[1], df2_lon[1]))

    df1_lat = df1_lat.reshape(-1, 1)
    df1_lon = df1_lon.reshape(-1, 1)
    df2_lat = df2_lat.reshape(1, -1)
    df2_lon = df2_lon.reshape(1, -1)
    # Calculate distance
    distance = haversine_numpy(df1_lat, df1_lon, df2_lat, df2_lon)

    return distance
