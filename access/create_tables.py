import pandas as pd
import geopandas as gpd
from db_access import get_db_connection_for_db
import json

def create_table_from_geojson(geojson_path, table_name, conn):
    """Read GeoJSON and create PostGIS table with appropriate schema"""
    # Load GeoJSON file
    gdf = gpd.read_file(geojson_path)
    
    # Ensure CRS is set (WGS84 / EPSG:4326)
    if gdf.crs is None:
        gdf.set_crs('EPSG:4326', inplace=True)
    elif gdf.crs.to_string() != 'EPSG:4326':
        gdf.to_crs('EPSG:4326', inplace=True)
    
    # Clean column names (lowercase, replace spaces and special chars)
    gdf.columns = [col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_') 
                   for col in gdf.columns]
    
    # Generate CREATE TABLE SQL (excluding geometry column - handled separately)
    columns = []
    for col in gdf.columns:
        if col == 'geometry':
            continue  # Skip geometry, handled separately
        
        dtype = gdf[col].dtype
        
        if dtype == 'int64':
            sql_type = 'INTEGER'
        elif dtype == 'float64':
            sql_type = 'DOUBLE PRECISION'
        elif dtype == 'bool':
            sql_type = 'BOOLEAN'
        elif dtype == 'object':
            # Check if it's a datetime-like string
            if gdf[col].dtype == 'object' and 'date' in col.lower():
                sql_type = 'TIMESTAMP'
            else:
                sql_type = 'TEXT'
        else:
            sql_type = 'TEXT'
        
        columns.append(f'"{col}" {sql_type}')
    
    # Enable PostGIS extension
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        conn.commit()
    
    # Create table with PostGIS geometry column
    # Use GEOMETRY(GEOMETRY, 4326) to accept both Polygon and MultiPolygon
    with conn.cursor() as cur:
        # Drop table if exists
        cur.execute(f'DROP TABLE IF EXISTS {table_name};')
        
        # Create table with flexible geometry type (accepts Polygon, MultiPolygon, etc.)
        # Use GEOMETRY without type constraint, then add SRID constraint
        create_table_sql = f'''
        CREATE TABLE {table_name} (
            id SERIAL PRIMARY KEY,
            {', '.join(columns)},
            geometry GEOMETRY
        );
        '''
        cur.execute(create_table_sql)
        
        # Add SRID constraint and ensure geometry is Polygon or MultiPolygon
        cur.execute(f'''
            ALTER TABLE {table_name} 
            ADD CONSTRAINT enforce_srid_geometry 
            CHECK (ST_SRID(geometry) = 4326);
            
            ALTER TABLE {table_name} 
            ADD CONSTRAINT enforce_geotype_geometry 
            CHECK (geometrytype(geometry) IN ('POLYGON', 'MULTIPOLYGON'));
        ''')
        conn.commit()
        cur.execute(create_table_sql)
        conn.commit()
        
        # Create spatial index for faster queries
        cur.execute(f'CREATE INDEX idx_{table_name}_geometry ON {table_name} USING GIST (geometry);')
        
        # Create indexes on commonly queried fields (if they exist)
        if 'zoningbasedistrict' in [col.lower() for col in gdf.columns]:
            cur.execute(f'CREATE INDEX idx_{table_name}_zoning ON {table_name}(zoningbasedistrict);')
        if 'councildistrict' in [col.lower() for col in gdf.columns]:
            cur.execute(f'CREATE INDEX idx_{table_name}_council ON {table_name}(councildistrict);')
        if 'zipcode' in [col.lower() for col in gdf.columns]:
            cur.execute(f'CREATE INDEX idx_{table_name}_zipcode ON {table_name}(zipcode);')
        
        conn.commit()
    
    print(f"Created PostGIS table: {table_name} with spatial indexes")

def create_table_from_csv(csv_path, table_name, conn):
    """Read CSV and create table with appropriate schema"""
    # Read CSV to infer schema
    df = pd.read_csv(csv_path, nrows=100)  # Sample first 100 rows
    
    # Generate CREATE TABLE SQL
    columns = []
    for col, dtype in df.dtypes.items():
        col_name = col.lower().replace(' ', '_').replace('(', '').replace(')', '')
        
        if dtype == 'int64':
            sql_type = 'INTEGER'
        elif dtype == 'float64':
            sql_type = 'DOUBLE PRECISION'
        elif dtype == 'bool':
            sql_type = 'BOOLEAN'
        else:
            sql_type = 'TEXT'
        
        columns.append(f'"{col_name}" {sql_type}')
    
    create_sql = f'''
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        {', '.join(columns)}
    );
    '''
    
    with conn.cursor() as cur:
        cur.execute(create_sql)
        conn.commit()
    
    print(f"Created table: {table_name}")
def create_aws_user_table():
    conn = get_db_connection_for_db('atrium_census')
    """Create AWS user table"""
    create_table_sql = f'''
    CREATE TABLE IF NOT EXISTS aws_user (
        id TEXT PRIMARY KEY,
        email TEXT NOT NULL,
        user_type TEXT NOT NULL,
        organization TEXT,
        neighborhood TEXT,
        other_specify TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    '''
    with conn.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS aws_user;')
        cur.execute(create_table_sql)
        conn.commit()
        print(f"Created AWS user table")
    conn.close()

def create_project_table():
    conn = get_db_connection_for_db('atrium_census')

    create_table_sql = '''
    CREATE TABLE IF NOT EXISTS project (
        id TEXT PRIMARY KEY,
        owner_id TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT,
        plots TEXT[] NOT NULL DEFAULT '{}',
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    '''

    with conn.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS project CASCADE;')
        cur.execute(create_table_sql)
        conn.commit()
        print("Created project table")
    conn.close()


def create_project_members_table():
    conn = get_db_connection_for_db('atrium_census')

    create_table_sql = '''
    CREATE TABLE IF NOT EXISTS project_members (
        project_id TEXT NOT NULL,
        member_id TEXT NOT NULL,
        role TEXT DEFAULT 'viewer',
        PRIMARY KEY (project_id, member_id),
        FOREIGN KEY (project_id) REFERENCES project(id) ON DELETE CASCADE
    );
    '''

    with conn.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS project_members;')
        cur.execute(create_table_sql)
        conn.commit()
        print("Created project members table")
    conn.close()


# Example usage
if __name__ == "__main__":
    create_project_table()
    '''
    import os
    
    conn = get_db_connection()
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create tables for your CSV files
    '''
    '''
    create_table_from_csv('data/PPR_Properties.csv', 'ppr_properties', conn)
    create_table_from_csv('data/PPR_Trails.csv', 'ppr_trails', conn)
    create_table_from_csv('data/Land_Use (1).csv', 'land_use', conn)
    '''
    '''

    # Create table for GeoJSON files
    geojson_path = os.path.normpath(os.path.join(current_dir, "../data/Vacant_Indicators_Land.geojson"))
    if os.path.exists(geojson_path):
        create_table_from_geojson(geojson_path, 'vacant_indicators_land', conn)
    else:
        print(f" GeoJSON file not found: {geojson_path}")
    
    conn.close()
    '''

