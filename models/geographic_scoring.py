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

def score_parks_proximity(lat: float, lon: float, parks_df: pd.DataFrame, 
                         gardens_df: pd.DataFrame, program_sites_df: pd.DataFrame) -> float:
    min_dist = float('inf')
    park_count_500m = 0
    park_count_1000m = 0
    total_acreage_1000m = 0
    
    for idx, row in program_sites_df.iterrows():
        if pd.notna(row.get('Y')) and pd.notna(row.get('X')):
            site_lat, site_lon = row['Y'], row['X']
            dist = calculate_distance(lat, lon, site_lat, site_lon)
            if dist < min_dist:
                min_dist = dist
            if dist <= 500:
                park_count_500m += 1
            if dist <= 1000:
                park_count_1000m += 1
    
    for idx, row in gardens_df.iterrows():
        if pd.notna(row.get('Y')) and pd.notna(row.get('X')):
            garden_lat, garden_lon = row['Y'], row['X']
            dist = calculate_distance(lat, lon, garden_lat, garden_lon)
            if dist < min_dist:
                min_dist = dist
            if dist <= 500:
                park_count_500m += 1
            if dist <= 1000:
                park_count_1000m += 1
    
    for idx, row in parks_df.iterrows():
        zip_code = str(row.get('zip_code', ''))
        if zip_code and zip_code.isdigit():
            acreage = row.get('acreage', 0)
            if pd.notna(acreage) and acreage > 0:
                park_count_1000m += 0.3
                total_acreage_1000m += acreage * 0.3
    
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
    tree_count_200m = 0
    tree_count_500m = 0
    total_dbh_500m = 0
    
    for idx, row in trees_df.iterrows():
        if pd.notna(row.get('loc_y')) and pd.notna(row.get('loc_x')):
            tree_lat, tree_lon = row['loc_y'], row['loc_x']
            dist = calculate_distance(lat, lon, tree_lat, tree_lon)
            
            if dist <= 200:
                tree_count_200m += 1
                dbh = row.get('tree_dbh', 0)
                if pd.notna(dbh):
                    total_dbh_500m += dbh
            elif dist <= 500:
                tree_count_500m += 1
                dbh = row.get('tree_dbh', 0)
                if pd.notna(dbh):
                    total_dbh_500m += dbh
    
    density_score = min(5, tree_count_200m / 10)
    coverage_score = min(3, tree_count_500m / 20)
    maturity_score = min(2, total_dbh_500m / 500)
    
    return min(10, density_score + coverage_score + maturity_score)

def score_environmental_friendliness(lat: float, lon: float, parks_df: pd.DataFrame, 
                                    trails_df: pd.DataFrame, trees_df: pd.DataFrame, 
                                    gardens_df: pd.DataFrame, program_sites_df: pd.DataFrame) -> float:
    park_score = score_parks_proximity(lat, lon, parks_df, gardens_df, program_sites_df)
    trail_score = score_trails_proximity(lat, lon, trails_df)
    tree_score = score_tree_density(lat, lon, trees_df)
    
    weighted_score = (park_score * 0.4) + (trail_score * 0.3) + (tree_score * 0.3)
    return round(weighted_score, 2)

def score_recreational_access(lat: float, lon: float, parks_df: pd.DataFrame, 
                              trails_df: pd.DataFrame, program_sites_df: pd.DataFrame) -> float:
    park_score = score_parks_proximity(lat, lon, parks_df, pd.DataFrame(), program_sites_df)
    trail_score = score_trails_proximity(lat, lon, trails_df)
    
    weighted_score = (park_score * 0.6) + (trail_score * 0.4)
    return round(weighted_score, 2)

def score_green_space_quality(lat: float, lon: float, parks_df: pd.DataFrame, 
                              trees_df: pd.DataFrame, gardens_df: pd.DataFrame, 
                              program_sites_df: pd.DataFrame) -> float:
    park_score = score_parks_proximity(lat, lon, parks_df, gardens_df, program_sites_df)
    tree_score = score_tree_density(lat, lon, trees_df)
    
    garden_count_500m = 0
    for idx, row in gardens_df.iterrows():
        if pd.notna(row.get('Y')) and pd.notna(row.get('X')):
            garden_lat, garden_lon = row['Y'], row['X']
            dist = calculate_distance(lat, lon, garden_lat, garden_lon)
            if dist <= 500:
                garden_count_500m += 1
    
    garden_score = min(3, garden_count_500m * 1.5)
    
    weighted_score = (park_score * 0.5) + (tree_score * 0.4) + (garden_score * 0.1)
    return round(weighted_score, 2)

def score_walkability(lat: float, lon: float, parks_df: pd.DataFrame, 
                     trails_df: pd.DataFrame, gardens_df: pd.DataFrame, 
                     program_sites_df: pd.DataFrame) -> float:
    park_count_800m = 0
    trail_count_800m = 0
    garden_count_800m = 0
    
    for idx, row in program_sites_df.iterrows():
        if pd.notna(row.get('Y')) and pd.notna(row.get('X')):
            site_lat, site_lon = row['Y'], row['X']
            dist = calculate_distance(lat, lon, site_lat, site_lon)
            if dist <= 800:
                park_count_800m += 1
    
    for idx, row in gardens_df.iterrows():
        if pd.notna(row.get('Y')) and pd.notna(row.get('X')):
            garden_lat, garden_lon = row['Y'], row['X']
            dist = calculate_distance(lat, lon, garden_lat, garden_lon)
            if dist <= 800:
                garden_count_800m += 1
    
    major_trails = trails_df[trails_df.get('TRAIL_TYPE', '') == 'MAJOR']
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

