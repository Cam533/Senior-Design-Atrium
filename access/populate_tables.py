import pandas as pd
import geopandas as gpd
from db_access import get_db_connection, get_db_engine
from psycopg2.extras import execute_values
import os

def populate_table_from_geojson(geojson_path, table_name, engine):
    """Load GeoJSON data into PostGIS RDS table using GeoPandas"""
    # Load GeoJSON file with GeoPandas (handles structure correctly)
    gdf = gpd.read_file(geojson_path)
    
    # Ensure CRS is set (WGS84 / EPSG:4326)
    if gdf.crs is None:
        gdf.set_crs('EPSG:4326', inplace=True)
    elif gdf.crs.to_string() != 'EPSG:4326':
        gdf.to_crs('EPSG:4326', inplace=True)
    
    # Clean column names (lowercase, replace spaces and special chars)
    gdf.columns = [col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_') 
                   for col in gdf.columns]
    
    # Ensure geometry column is named 'geometry'
    if 'geometry' not in gdf.columns:
        # If geometry column has different name, rename it
        geom_col = [col for col in gdf.columns if gdf[col].dtype == 'geometry'][0]
        gdf.rename(columns={geom_col: 'geometry'}, inplace=True)
    
    print(f"Loading {len(gdf)} features from {geojson_path}...")
    
    # Use GeoPandas to_postgis() method - handles geometry insertion correctly
    # Requires SQLAlchemy engine (not psycopg2 connection)
    gdf.to_postgis(
        table_name,
        engine,  # SQLAlchemy engine, not psycopg2 connection
        if_exists='append',  # Append to existing table
        index=False,  # Don't include pandas index
        chunksize=1000  # Insert in chunks
    )
    
    print(f"Populated table: {table_name} with {len(gdf)} rows")

def populate_table_from_csv(csv_path, table_name, conn, chunk_size=1000):
    """Load CSV data into RDS table in chunks"""
    df = pd.read_csv(csv_path)
    
    # Clean column names
    df.columns = [col.lower().replace(' ', '_').replace('(', '').replace(')', '') 
                  for col in df.columns]
    
    # Convert to list of tuples
    values = [tuple(row) for row in df.values]
    columns = ','.join([f'"{col}"' for col in df.columns])
    
    # Insert in chunks
    with conn.cursor() as cur:
        for i in range(0, len(values), chunk_size):
            chunk = values[i:i+chunk_size]
            placeholders = ','.join(['%s'] * len(df.columns))
            
            insert_sql = f'''
            INSERT INTO {table_name} ({columns})
            VALUES {placeholders}
            ON CONFLICT DO NOTHING;
            '''
            
            # Use execute_values for better performance
            execute_values(cur, 
                f'INSERT INTO {table_name} ({columns}) VALUES %s',
                chunk)
            
            conn.commit()
            print(f"  Inserted {min(i+chunk_size, len(values))}/{len(values)} rows")
    
    print(f"Populated table: {table_name} with {len(df)} rows")

if __name__ == "__main__":
    import os
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # For CSV tables, use psycopg2 connection
    conn = get_db_connection()
    # populate_table_from_csv('data/PPR_Properties.csv', 'ppr_properties', conn)
    # populate_table_from_csv('data/PPR_Trails.csv', 'ppr_trails', conn)
    # populate_table_from_csv('data/Land_Use (1).csv', 'land_use', conn)
    conn.close()
    
    # For GeoJSON tables, use SQLAlchemy engine (required by to_postgis)
    engine = get_db_engine()
    geojson_path = os.path.normpath(os.path.join(current_dir, "../data/Vacant_Indicators_Land.geojson"))
    if os.path.exists(geojson_path):
        populate_table_from_geojson(geojson_path, 'vacant_indicators_land', engine)
    else:
        print(f"GeoJSON file not found: {geojson_path}")
    
    engine.dispose()  # Close SQLAlchemy engine