import pandas as pd
import numpy as np
from geopy.distance import geodesic
from typing import Dict
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

_parks_df = None
_trails_df = None
_trees_df = None
_gardens_df = None
_program_sites_df = None
_transit_stops_df = None
_bike_network_df = None
_complete_streets_df = None
_mini_city_halls_df = None


def _load_data():
    global _parks_df, _trails_df, _trees_df, _gardens_df, _program_sites_df
    global _transit_stops_df, _bike_network_df, _complete_streets_df, _mini_city_halls_df
    if _parks_df is None:
        _parks_df = pd.read_csv(os.path.join(DATA_DIR, "PPR_Properties.csv"))
        _trails_df = pd.read_csv(os.path.join(DATA_DIR, "PPR_Trails.csv"))
        _trees_df = pd.read_csv(os.path.join(DATA_DIR, "ppr_tree_inventory_2024.csv"))
        _gardens_df = pd.read_csv(os.path.join(DATA_DIR, "Registered_Community_Gardens.csv"))
        _program_sites_df = pd.read_csv(os.path.join(DATA_DIR, "PPR_Program_Sites.csv"))
        _transit_stops_df = pd.read_csv(os.path.join(DATA_DIR, "Transit_Stops_(Spring_2025).csv"))
        _bike_network_df = pd.read_csv(os.path.join(DATA_DIR, "Bike_Network.csv"))
        _complete_streets_df = pd.read_csv(os.path.join(DATA_DIR, "CompleteStreets.csv"))
        _mini_city_halls_df = pd.read_csv(os.path.join(DATA_DIR, "mini_city_halls.csv"))
    return (_parks_df, _trails_df, _trees_df, _gardens_df, _program_sites_df,
            _transit_stops_df, _bike_network_df, _complete_streets_df, _mini_city_halls_df)


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return float('inf')
    return geodesic((lat1, lon1), (lat2, lon2)).meters


def score_parks_proximity(
    lat: float, lon: float, parks_df: pd.DataFrame,
    gardens_df: pd.DataFrame, program_sites_df: pd.DataFrame
) -> float:
    min_dist = float('inf')
    park_count_500m = 0
    park_count_1000m = 0
    total_acreage_1000m = 0
    
    if 'Y' in program_sites_df.columns and 'X' in program_sites_df.columns:
        sites_valid = program_sites_df[(program_sites_df['Y'].notna()) & (program_sites_df['X'].notna())]
        if len(sites_valid) > 0:
            sites_valid = sites_valid.copy()
            sites_valid['dist'] = sites_valid.apply(
                lambda row: calculate_distance(lat, lon, row['Y'], row['X']), 
                axis=1
            )
            min_dist = min(min_dist, sites_valid['dist'].min())
            park_count_500m += len(sites_valid[sites_valid['dist'] <= 500])
            park_count_1000m += len(sites_valid[sites_valid['dist'] <= 1000])
    
    if 'Y' in gardens_df.columns and 'X' in gardens_df.columns:
        gardens_valid = gardens_df[(gardens_df['Y'].notna()) & (gardens_df['X'].notna())]
        if len(gardens_valid) > 0:
            gardens_valid = gardens_valid.copy()
            gardens_valid['dist'] = gardens_valid.apply(
                lambda row: calculate_distance(lat, lon, row['Y'], row['X']), 
                axis=1
            )
            min_dist = min(min_dist, gardens_valid['dist'].min())
            park_count_500m += len(gardens_valid[gardens_valid['dist'] <= 500])
            park_count_1000m += len(gardens_valid[gardens_valid['dist'] <= 1000])
    
    
    if min_dist == float('inf'):
        return 0.0
    
    proximity_score = max(0, 10 - (min_dist / 200))
    density_score = min(5, park_count_500m * 1.5) + min(5, park_count_1000m * 0.5)
    size_score = min(5, total_acreage_1000m / 10)
    
    return float(min(10, proximity_score + density_score + size_score))


def score_trails_proximity(lat: float, lon: float, trails_df: pd.DataFrame) -> float:
    return 0.0


def score_tree_density(lat: float, lon: float, trees_df: pd.DataFrame) -> float:
    trees_valid = trees_df[(trees_df['loc_y'].notna()) & (trees_df['loc_x'].notna())].copy()
    
    if len(trees_valid) == 0:
        return 0.0
    
    lat_diff = np.radians(trees_valid['loc_y'].values - lat)
    lon_diff = np.radians(trees_valid['loc_x'].values - lon)
    
    a = np.sin(lat_diff/2)**2 + np.cos(np.radians(lat)) * np.cos(np.radians(trees_valid['loc_y'].values)) * np.sin(lon_diff/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    dists = 6371000 * c
    
    trees_valid['dist'] = dists
    
    trees_200m = trees_valid[trees_valid['dist'] <= 200]
    trees_500m = trees_valid[trees_valid['dist'] <= 500]
    
    tree_count_200m = len(trees_200m)
    tree_count_500m = len(trees_500m)
    
    dbh_500m = trees_500m['tree_dbh'].fillna(0)
    total_dbh_500m = dbh_500m.sum()
    
    density_score = min(5, tree_count_200m / 10)
    coverage_score = min(3, tree_count_500m / 20)
    maturity_score = min(2, total_dbh_500m / 500)
    
    return float(min(10, density_score + coverage_score + maturity_score))


def score_environmental_friendliness(
    lat: float, lon: float, parks_df: pd.DataFrame,
    trails_df: pd.DataFrame, trees_df: pd.DataFrame,
    gardens_df: pd.DataFrame, program_sites_df: pd.DataFrame
) -> float:
    tree_score = score_tree_density(lat, lon, trees_df)
    
    garden_count_500m = 0
    if 'Y' in gardens_df.columns and 'X' in gardens_df.columns:
        gardens_valid = gardens_df[(gardens_df['Y'].notna()) & (gardens_df['X'].notna())]
        if len(gardens_valid) > 0:
            gardens_valid = gardens_valid.copy()
            gardens_valid['dist'] = gardens_valid.apply(
                lambda row: calculate_distance(lat, lon, row['Y'], row['X']),
                axis=1
            )
            garden_count_500m = len(gardens_valid[gardens_valid['dist'] <= 500])
    
    garden_score = min(3, garden_count_500m * 1.5)
    
    weighted_score = (tree_score * 0.7) + (garden_score * 0.3)
    return round(weighted_score, 2)


def score_recreational_access(
    lat: float, lon: float, parks_df: pd.DataFrame,
    trails_df: pd.DataFrame, program_sites_df: pd.DataFrame
) -> float:
    park_score = score_parks_proximity(lat, lon, parks_df, pd.DataFrame(), program_sites_df)
    return round(park_score, 2)


def score_green_space_quality(
    lat: float, lon: float, parks_df: pd.DataFrame,
    trees_df: pd.DataFrame, gardens_df: pd.DataFrame,
    program_sites_df: pd.DataFrame
) -> float:
    park_score = score_parks_proximity(lat, lon, parks_df, gardens_df, program_sites_df)
    tree_score = score_tree_density(lat, lon, trees_df)
    
    garden_count_500m = 0
    if 'Y' in gardens_df.columns and 'X' in gardens_df.columns:
        gardens_valid = gardens_df[(gardens_df['Y'].notna()) & (gardens_df['X'].notna())]
        if len(gardens_valid) > 0:
            gardens_valid = gardens_valid.copy()
            gardens_valid['dist'] = gardens_valid.apply(
                lambda row: calculate_distance(lat, lon, row['Y'], row['X']), 
                axis=1
            )
            garden_count_500m = len(gardens_valid[gardens_valid['dist'] <= 500])
    
    garden_score = min(3, garden_count_500m * 1.5)
    
    weighted_score = (tree_score * 0.7) + (garden_score * 0.3)
    return round(weighted_score, 2)


def score_walkability(
    lat: float, lon: float, parks_df: pd.DataFrame,
    trails_df: pd.DataFrame, gardens_df: pd.DataFrame,
    program_sites_df: pd.DataFrame, complete_streets_df: pd.DataFrame,
    transit_stops_df: pd.DataFrame
) -> float:
    park_count_400m = 0
    park_count_800m = 0
    if 'Y' in program_sites_df.columns and 'X' in program_sites_df.columns:
        sites_valid = program_sites_df[(program_sites_df['Y'].notna()) & (program_sites_df['X'].notna())]
        if len(sites_valid) > 0:
            sites_valid = sites_valid.copy()
            sites_valid['dist'] = sites_valid.apply(
                lambda row: calculate_distance(lat, lon, row['Y'], row['X']),
                axis=1
            )
            park_count_400m = len(sites_valid[sites_valid['dist'] <= 400])
            park_count_800m = len(sites_valid[sites_valid['dist'] <= 800])
    
    garden_count_800m = 0
    if 'Y' in gardens_df.columns and 'X' in gardens_df.columns:
        gardens_valid = gardens_df[(gardens_df['Y'].notna()) & (gardens_df['X'].notna())]
        if len(gardens_valid) > 0:
            gardens_valid = gardens_valid.copy()
            gardens_valid['dist'] = gardens_valid.apply(
                lambda row: calculate_distance(lat, lon, row['Y'], row['X']),
                axis=1
            )
            garden_count_800m = len(gardens_valid[gardens_valid['dist'] <= 800])
    
    transit_count_400m = 0
    if 'Lat' in transit_stops_df.columns and 'Lon' in transit_stops_df.columns:
        stops_valid = transit_stops_df[(transit_stops_df['Lat'].notna()) & (transit_stops_df['Lon'].notna())]
        if len(stops_valid) > 0:
            stops_valid = stops_valid.copy()
            stops_valid['dist'] = stops_valid.apply(
                lambda row: calculate_distance(lat, lon, row['Lat'], row['Lon']),
                axis=1
            )
            transit_count_400m = len(stops_valid[stops_valid['dist'] <= 400])
    
    destination_score = min(6, park_count_400m * 2.0 + park_count_800m * 0.5 + garden_count_800m * 1.5)
    transit_access_score = min(4, transit_count_400m * 0.3)
    
    walkability_score = destination_score + transit_access_score
    
    return round(min(10, walkability_score), 2)


def score_transit_accessibility(lat: float, lon: float, transit_stops_df: pd.DataFrame) -> float:
    if 'Lat' not in transit_stops_df.columns or 'Lon' not in transit_stops_df.columns:
        return 0.0
    
    stops_valid = transit_stops_df[(transit_stops_df['Lat'].notna()) & (transit_stops_df['Lon'].notna())]
    if len(stops_valid) == 0:
        return 0.0
    
    stops_valid = stops_valid.copy()
    stops_valid['dist'] = stops_valid.apply(
        lambda row: calculate_distance(lat, lon, row['Lat'], row['Lon']),
        axis=1
    )
    
    min_dist = stops_valid['dist'].min()
    stops_400m = len(stops_valid[stops_valid['dist'] <= 400])
    stops_800m = len(stops_valid[stops_valid['dist'] <= 800])
    
    proximity_score = max(0, 5 - (min_dist / 200))
    density_score = min(5, stops_400m * 0.5) + min(5, stops_800m * 0.2)
    
    return float(min(10, proximity_score + density_score))


def score_bike_infrastructure(lat: float, lon: float, bike_network_df: pd.DataFrame, 
                              complete_streets_df: pd.DataFrame) -> float:
    bike_score = 0.0
    
    if 'STREETNAME' in bike_network_df.columns and 'TYPE' in bike_network_df.columns:
        separated_lanes = bike_network_df[bike_network_df['TYPE'].str.contains('Separated', na=False)]
        protected_lanes = bike_network_df[bike_network_df['TYPE'].str.contains('Protected', na=False)]
        bike_score += min(5, len(separated_lanes) * 0.1) + min(5, len(protected_lanes) * 0.15)
    
    if 'BIKENETWOR' in complete_streets_df.columns:
        bike_facilities = complete_streets_df[complete_streets_df['BIKENETWOR'].notna()]
        bike_score += min(5, len(bike_facilities) * 0.01)
    
    return min(10, bike_score)


def calculate_all_scores(lat: float, lon: float) -> Dict[str, float]:
    (parks_df, trails_df, trees_df, gardens_df, program_sites_df,
     transit_stops_df, bike_network_df, complete_streets_df, mini_city_halls_df) = _load_data()
    
    environmental = score_environmental_friendliness(lat, lon, parks_df, trails_df, trees_df, gardens_df, program_sites_df)
    recreational = score_recreational_access(lat, lon, parks_df, trails_df, program_sites_df)
    transit = score_transit_accessibility(lat, lon, transit_stops_df)
    walkability = score_walkability(lat, lon, parks_df, trails_df, gardens_df, program_sites_df, complete_streets_df, transit_stops_df)
    
    return {
        'environmental_score': environmental,
        'recreational_score': recreational,
        'transit_score': transit,
        'walkability_score': walkability
    }


def score_location(lat: float, lon: float) -> Dict[str, float]:
    return calculate_all_scores(lat, lon)

