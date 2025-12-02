import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from geographic_scoring import score_location

def add_scores_to_csv(input_file: str, output_file: str, lat_col: str = 'lat', lon_col: str = 'lon'):
    df = pd.read_csv(input_file)
    
    if lat_col not in df.columns or lon_col not in df.columns:
        print(f"Error: Columns '{lat_col}' and '{lon_col}' must exist in the CSV")
        return
    
    scores_list = []
    for idx, row in df.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]
        
        if pd.notna(lat) and pd.notna(lon):
            scores = score_location(lat, lon)
            scores_list.append(scores)
        else:
            scores_list.append({
                'environmental_score': None,
                'recreational_score': None,
                'green_space_score': None,
                'walkability_score': None
            })
    
    scores_df = pd.DataFrame(scores_list)
    result_df = pd.concat([df, scores_df], axis=1)
    result_df.to_csv(output_file, index=False)
    print(f"Added scores to {len(df)} locations. Saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python add_scores_to_locations.py <input_file> <output_file> [lat_col] [lon_col]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    lat_col = sys.argv[3] if len(sys.argv) > 3 else 'lat'
    lon_col = sys.argv[4] if len(sys.argv) > 4 else 'lon'
    
    add_scores_to_csv(input_file, output_file, lat_col, lon_col)

