import pandas as pd
import sys
import os
import json
from tqdm import tqdm
import logging
import geopandas as gpd
from shapely.geometry import shape
from multiprocessing import Pool, cpu_count
from functools import partial
logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from geographic_scoring import score_location


def process_single_feature(feature):
    """Process a single feature to calculate scores."""
    try:
        geom = shape(feature["geometry"])     # convert GeoJSON → Shapely
        
        # Get centroid to extract lat/lon
        centroid = geom.centroid
        lon = centroid.x  # longitude (x coordinate)
        lat = centroid.y  # latitude (y coordinate)
        
        # Calculate scores using the centroid coordinates
        scores = score_location(lat=lat, lon=lon)
        
        # Ensure properties exist
        if "properties" not in feature:
            feature["properties"] = {}
        
        # Update feature properties with scores
        feature["properties"].update(scores)
        
        return feature, True
    except Exception as e:
        logging.warning(f"Error processing feature: {e}")
        # Add None scores on error
        if "properties" not in feature:
            feature["properties"] = {}
        feature["properties"].update({
            'environmental_score': None,
            'recreational_score': None,
            'green_space_score': None,
            'walkability_score': None
        })
        return feature, False

def add_scores_to_geojson(input_geojson_file: str, output_geojson_file: str, num_workers: int = None):
    """
    Read a GeoJSON file, calculate scores for each feature based on its geometry,
    and save the result with scores added to each feature's properties.
    
    Args:
        input_geojson_file: Path to input GeoJSON file
        output_geojson_file: Path to output GeoJSON file
        num_workers: Number of parallel workers (defaults to CPU count - 1)
    """
    # Read GeoJSON file
    with open(input_geojson_file, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    
    if geojson_data.get('type') != 'FeatureCollection':
        print(f"Error: Expected FeatureCollection, got {geojson_data.get('type')}")
        return
    
    features = geojson_data['features']
    print(f"Processing {len(features)} features...")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Use all but one CPU core
    
    print(f"Using {num_workers} parallel workers...")
    
    # Process features in parallel
    processed_count = 0
    if num_workers > 1 and len(features) > 100:  # Only use multiprocessing for large datasets
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_feature, features),
                total=len(features),
                desc="Processing features"
            ))
        
        # Update features with results
        for i, (feature, success) in enumerate(results):
            features[i] = feature
            if success:
                processed_count += 1
    else:
        # Sequential processing for small datasets
        for feature in tqdm(features, desc="Processing features"):
            feature, success = process_single_feature(feature)
            if success:
                processed_count += 1
    
    # Save updated GeoJSON
    print(f"Saving results to {output_geojson_file}...")
    with open(output_geojson_file, 'w', encoding='utf-8') as f:
        json.dump(geojson_data, f, indent=2)
    
    print(f"Added scores to {processed_count}/{len(features)} features. Saved to {output_geojson_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  For GeoJSON: python add_scores_to_locations_copy.py <input.geojson> <output.geojson> [num_workers]")
        print("  Example: python add_scores_to_locations_copy.py input.geojson output.geojson 8")
        print("  (num_workers is optional, defaults to CPU count - 1)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    num_workers = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    # Check if input is GeoJSON or CSV
    if input_file.lower().endswith('.geojson'):
        add_scores_to_geojson(input_file, output_file, num_workers)
    else:
        print("Input file must be a GeoJSON file")
        sys.exit(1)

