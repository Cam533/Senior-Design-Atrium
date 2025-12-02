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


def _load_data():
    global _parks_df, _trails_df, _trees_df, _gardens_df, _program_sites_df
    if _parks_df is None:
        _parks_df = pd.read_csv(os.path.join(DATA_DIR, "PPR_Properties.csv"))
        _trails_df = pd.read_csv(os.path.join(DATA_DIR, "PPR_Trails.csv"))
        _trees_df = pd.read_csv(os.path.join(DATA_DIR, "ppr_tree_inventory_2024.csv"))
        _gardens_df = pd.read_csv(os.path.join(DATA_DIR, "Registered_Community_Gardens.csv"))
        _program_sites_df = pd.read_csv(os.path.join(DATA_DIR, "PPR_Program_Sites.csv"))
    return _parks_df, _trails_df, _trees_df, _gardens_df, _program_sites_df


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
    
    if 'acreage' in parks_df.columns:
        parks_with_acreage = parks_df[(parks_df['acreage'].notna()) & (parks_df['acreage'] > 0)]
        if len(parks_with_acreage) > 0:
            park_count_1000m += len(parks_with_acreage) * 0.3
            total_acreage_1000m += parks_with_acreage['acreage'].sum() * 0.3
    
    if min_dist == float('inf'):
        return 0.0
    
    proximity_score = max(0, 10 - (min_dist / 200))
    density_score = min(5, park_count_500m * 1.5) + min(5, park_count_1000m * 0.5)
    size_score = min(5, total_acreage_1000m / 10)
    
    return min(10, proximity_score + density_score + size_score)


def score_trails_proximity(lat: float, lon: float, trails_df: pd.DataFrame) -> float:
    trail_length_500m = 0
    trail_count_1000m = 0
    major_trail_count = 0
    
    for idx, row in trails_df.iterrows():
        if pd.isna(row.get('MILES')):
            continue
        
        trail_miles = row.get('MILES', 0)
        trail_type = row.get('TRAIL_TYPE', '')
        
        if trail_type == 'MAJOR':
            major_trail_count += 1
            trail_length_500m += trail_miles * 0.5
            trail_count_1000m += 1
    
    proximity_score = min(5, major_trail_count * 1.0)
    length_score = min(5, trail_length_500m * 2)
    count_score = min(5, trail_count_1000m * 0.3)
    
    return min(10, proximity_score + length_score + count_score)


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
    
    return min(10, density_score + coverage_score + maturity_score)


def score_environmental_friendliness(
    lat: float, lon: float, parks_df: pd.DataFrame,
    trails_df: pd.DataFrame, trees_df: pd.DataFrame,
    gardens_df: pd.DataFrame, program_sites_df: pd.DataFrame
) -> float:
    park_score = score_parks_proximity(lat, lon, parks_df, gardens_df, program_sites_df)
    trail_score = score_trails_proximity(lat, lon, trails_df)
    tree_score = score_tree_density(lat, lon, trees_df)
    
    weighted_score = (park_score * 0.4) + (trail_score * 0.3) + (tree_score * 0.3)
    return round(weighted_score, 2)


def score_recreational_access(
    lat: float, lon: float, parks_df: pd.DataFrame,
    trails_df: pd.DataFrame, program_sites_df: pd.DataFrame
) -> float:
    park_score = score_parks_proximity(lat, lon, parks_df, pd.DataFrame(), program_sites_df)
    trail_score = score_trails_proximity(lat, lon, trails_df)
    
    weighted_score = (park_score * 0.6) + (trail_score * 0.4)
    return round(weighted_score, 2)


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
    
    weighted_score = (park_score * 0.5) + (tree_score * 0.4) + (garden_score * 0.1)
    return round(weighted_score, 2)


def score_walkability(
    lat: float, lon: float, parks_df: pd.DataFrame,
    trails_df: pd.DataFrame, gardens_df: pd.DataFrame,
    program_sites_df: pd.DataFrame
) -> float:
    park_count_800m = 0
    if 'Y' in program_sites_df.columns and 'X' in program_sites_df.columns:
        sites_valid = program_sites_df[(program_sites_df['Y'].notna()) & (program_sites_df['X'].notna())]
        if len(sites_valid) > 0:
            sites_valid = sites_valid.copy()
            sites_valid['dist'] = sites_valid.apply(
                lambda row: calculate_distance(lat, lon, row['Y'], row['X']), 
                axis=1
            )
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
    
    trail_count_800m = 0
    if 'TRAIL_TYPE' in trails_df.columns:
        major_trails = trails_df[trails_df['TRAIL_TYPE'] == 'MAJOR']
        trail_count_800m = len(major_trails) * 0.2
    
    total_destinations = park_count_800m + trail_count_800m + garden_count_800m
    walkability_score = min(10, total_destinations * 1.5)
    
    return round(walkability_score, 2)


def calculate_all_scores(lat: float, lon: float) -> Dict[str, float]:
    parks_df, trails_df, trees_df, gardens_df, program_sites_df = _load_data()
    
    environmental = score_environmental_friendliness(lat, lon, parks_df, trails_df, trees_df, gardens_df, program_sites_df)
    recreational = score_recreational_access(lat, lon, parks_df, trails_df, program_sites_df)
    green_quality = score_green_space_quality(lat, lon, parks_df, trees_df, gardens_df, program_sites_df)
    walkability = score_walkability(lat, lon, parks_df, trails_df, gardens_df, program_sites_df)
    
    return {
        'environmental_score': environmental,
        'recreational_score': recreational,
        'green_space_score': green_quality,
        'walkability_score': walkability
    }


def score_location(lat: float, lon: float) -> Dict[str, float]:
    return calculate_all_scores(lat, lon)

