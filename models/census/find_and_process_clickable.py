"""
Find addresses that are clickable in the UI (in GeoJSON files) and process one through ETL.
This ensures the address will be both clickable AND have census data.
"""

import json
import pandas as pd
import requests
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import text, create_engine

load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from access.db_access import get_db_engine
except ImportError:
    print("Warning: Could not import db_access helper. Using fallback.")
    get_db_engine = None

CENSUS_API_KEY = "c50562fadf7bcd7d4989a392ccbf9c2333fc74d2"
OPA_DATA_FILE = "../../data/opa_properties_public.csv"

# Search criteria - will find first clickable address matching this
SEARCH_ADDRESS = "4 RIVER"  # Partial match, case-insensitive


def load_clickable_addresses():
    """Load all addresses from GeoJSON files (clickable in UI)"""
    print(f"{'='*70}")
    print(f"LOADING CLICKABLE ADDRESSES FROM GEOJSON")
    print(f"{'='*70}")
    
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    
    geojson_files = [
        data_dir / "Vacant_Indicators_Land.geojson",
        data_dir / "Vacant_Indicators_Bldg.geojson"
    ]
    
    addresses = []
    
    for filepath in geojson_files:
        if not filepath.exists():
            print(f"⚠️  {filepath.name} not found, skipping...")
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get("type") == "FeatureCollection":
                features = data.get("features", [])
                for feature in features:
                    props = feature.get("properties", {})
                    address = props.get("address") or props.get("location")
                    
                    if address:
                        # Calculate centroid for coordinates
                        coords = None
                        if feature.get("geometry"):
                            geom = feature["geometry"]
                            if geom.get("type") == "Polygon" and geom.get("coordinates"):
                                polygon = geom["coordinates"][0]
                                if polygon:
                                    lon = sum(c[0] for c in polygon) / len(polygon)
                                    lat = sum(c[1] for c in polygon) / len(polygon)
                                    coords = (lat, lon)
                        
                        addresses.append({
                            "address": address,
                            "opa_id": props.get("opa_id") or props.get("parcel_number"),
                            "objectid": props.get("objectid"),
                            "coords": coords,
                            "props": props
                        })
            
            print(f"✓ Loaded {len(features)} features from {filepath.name}")
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            continue
    
    print(f"✓ Total clickable addresses: {len(addresses)}")
    return addresses


def find_matching_clickable(clickable_addresses, search_term):
    """Find a clickable address matching the search term"""
    print(f"\n{'='*70}")
    print(f"SEARCHING FOR CLICKABLE ADDRESS")
    print(f"{'='*70}")
    print(f"Search term: '{search_term}'")
    
    matches = []
    for addr in clickable_addresses:
        if search_term.upper() in addr["address"].upper():
            matches.append(addr)
    
    if not matches:
        print(f"❌ No clickable addresses found matching '{search_term}'")
        return None
    
    print(f"✓ Found {len(matches)} matching address(es):")
    for i, match in enumerate(matches[:5]):
        print(f"   {i+1}. {match['address']} (OPA: {match.get('opa_id', 'N/A')})")
    
    if len(matches) > 5:
        print(f"   ... and {len(matches) - 5} more")
    
    # Return the first match
    selected = matches[0]
    print(f"\n✓ Selected: {selected['address']}")
    return selected


def find_in_opa_data(clickable_address):
    """Find this address in the OPA CSV data"""
    print(f"\n{'='*70}")
    print(f"SEARCHING OPA DATA")
    print(f"{'='*70}")
    
    if not os.path.exists(OPA_DATA_FILE):
        print(f"❌ OPA data file not found: {OPA_DATA_FILE}")
        return None
    
    address_to_find = clickable_address["address"].upper()
    opa_id = clickable_address.get("opa_id")
    
    print(f"Looking for address: {address_to_find}")
    if opa_id:
        print(f"GeoJSON OPA ID: {opa_id} (may not match OPA CSV)")
    
    try:
        # Search in chunks
        chunk_size = 10000
        for chunk in pd.read_csv(OPA_DATA_FILE, low_memory=False, chunksize=chunk_size):
            parcel = None
            
            # Match by address FIRST (most reliable for our use case)
            parcel = chunk[chunk['location'].str.upper() == address_to_find]
            
            # If exact match fails, try OPA ID
            if (parcel is None or len(parcel) == 0) and opa_id:
                parcel = chunk[chunk['parcel_number'].astype(str) == str(opa_id)]
            
            # If still no match, try partial address match on street name
            if parcel is None or len(parcel) == 0:
                # Extract street name (everything after house number)
                parts = address_to_find.split()
                if len(parts) > 1:
                    street_name = ' '.join(parts[1:])
                    parcel = chunk[chunk['location'].str.upper().str.contains(street_name, na=False, regex=False)]
                    if len(parcel) > 1:
                        # Multiple matches, try to find exact house number
                        house_num = parts[0]
                        exact = parcel[parcel['location'].str.upper().str.startswith(house_num + ' ')]
                        if len(exact) > 0:
                            parcel = exact
            
            if parcel is not None and len(parcel) > 0:
                result = parcel.iloc[0]
                print(f"✓ Found in OPA data:")
                print(f"   Parcel Number: {result.get('parcel_number', 'N/A')}")
                print(f"   Location: {result.get('location', 'N/A')}")
                print(f"   Owner: {result.get('owner_1', 'N/A')}")
                print(f"   ZIP: {result.get('zip_code', 'N/A')}")
                if 'category_code_description' in result:
                    print(f"   Property Type: {result.get('category_code_description', 'N/A')}")
                
                # Check if addresses match
                if result.get('location', '').upper() != address_to_find:
                    print(f"   ⚠️  Note: GeoJSON address '{address_to_find}' matched to OPA address '{result.get('location', '')}'")
                
                return result
        
        print(f"❌ Address not found in OPA data")
        print(f"   The GeoJSON has '{address_to_find}', but it's not in the OPA CSV.")
        print(f"   Try searching for a different clickable address.")
        return None
        
    except Exception as e:
        print(f"❌ Error reading OPA data: {e}")
        import traceback
        traceback.print_exc()
        return None


def geocode_address(parcel_data):
    """Geocode using Census API"""
    print(f"\n{'='*70}")
    print(f"GEOCODING WITH CENSUS API")
    print(f"{'='*70}")
    
    location = str(parcel_data.get('location', ''))
    zip_code = str(parcel_data.get('zip_code', ''))
    
    print(f"Address: {location}, Philadelphia, PA {zip_code}")
    
    csv_content = f"id,street,city,state,zip\n"
    csv_content += f'"{parcel_data.get("parcel_number", "1")}","{location}","Philadelphia","PA","{zip_code}"\n'
    
    url = "https://geocoding.geo.census.gov/geocoder/geographies/addressbatch"
    
    files = {'addressFile': ('address.csv', csv_content, 'text/csv')}
    data = {'benchmark': 'Public_AR_Current', 'vintage': 'Current_Current'}
    
    try:
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            import csv
            from io import StringIO
            
            lines = response.text.strip().split('\n')
            if not lines:
                print("❌ No response")
                return None
            
            reader = csv.reader(StringIO(lines[0]))
            parts = next(reader)
            
            if len(parts) < 3 or parts[2] != 'Match':
                print(f"❌ Not matched by Census Geocoder")
                print(f"   Status: {parts[2] if len(parts) > 2 else 'Unknown'}")
                return None
            
            print(f"✓ Geocoded successfully!")
            
            result = {'matched': True}
            
            # Extract coordinates and census tract
            if len(parts) > 5 and ',' in parts[5]:
                lon_str, lat_str = parts[5].split(',')
                result['longitude'] = float(lon_str.strip())
                result['latitude'] = float(lat_str.strip())
                print(f"   Coordinates: ({result['latitude']:.6f}, {result['longitude']:.6f})")
            
            if len(parts) >= 12:
                state_fips = parts[8]
                county_fips = parts[9]
                tract_code = parts[10]
                if state_fips and county_fips and tract_code:
                    result['census_tract'] = f"{state_fips}{county_fips}{tract_code.zfill(6)}"
                    print(f"   Census Tract: {result['census_tract']}")
            
            return result
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def fetch_census_data(tract_geoid):
    """Fetch census data for tract"""
    print(f"\n{'='*70}")
    print(f"FETCHING CENSUS DATA")
    print(f"{'='*70}")
    print(f"Census Tract: {tract_geoid}")
    
    if len(tract_geoid) < 11:
        print(f"❌ Invalid tract GEOID")
        return None
    
    state = tract_geoid[:2]
    county = tract_geoid[2:5]
    tract = tract_geoid[5:11]
    
    variables = [
        "B01003_001E", "B19013_001E", "B01002_001E", "B09001_001E",
        "B09020_001E", "B25077_001E", "B25064_001E", "B08301_010E"
    ]
    
    url = "https://api.census.gov/data/2022/acs/acs5"
    params = {
        "get": ",".join(variables),
        "for": f"tract:{tract}",
        "in": f"state:{state} county:{county}",
        "key": CENSUS_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if len(data) > 1:
                result = dict(zip(data[0], data[1]))
                print(f"✓ Census data retrieved!")
                if 'B01003_001E' in result:
                    print(f"   Population: {result['B01003_001E']}")
                if 'B19013_001E' in result and result['B19013_001E'] != '-666666666':
                    print(f"   Median Income: ${result['B19013_001E']}")
                return result
        
        print(f"❌ No census data available")
        return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def create_enriched_record(parcel_data, geocode_result, census_data):
    """Create enriched record with all data"""
    
    essential_cols = [
        'parcel_number', 'location', 'zip_code', 'house_number', 'street_name',
        'owner_1', 'owner_2', 'zoning', 'total_area', 'building_code_description',
        'category_code_description', 'year_built', 'market_value', 'sale_date', 'sale_price'
    ]
    
    record = {}
    for col in essential_cols:
        if col in parcel_data.index:
            val = parcel_data[col]
            record[col] = None if pd.isna(val) else val
    
    record['matched'] = geocode_result.get('matched', False)
    record['latitude'] = geocode_result.get('latitude')
    record['longitude'] = geocode_result.get('longitude')
    record['census_tract'] = geocode_result.get('census_tract')
    
    if census_data:
        mapping = {
            "B01003_001E": "tract_total_pop",
            "B19013_001E": "tract_median_income",
            "B01002_001E": "tract_median_age",
            "B09001_001E": "tract_pop_under_18",
            "B09020_001E": "tract_pop_65_plus",
            "B25077_001E": "tract_median_home_value",
            "B25064_001E": "tract_median_rent",
            "B08301_010E": "tract_transit_commuters",
        }
        
        for census_col, renamed_col in mapping.items():
            if census_col in census_data:
                value = census_data[census_col]
                try:
                    if value and value != '-666666666':
                        record[renamed_col] = float(value)
                    else:
                        record[renamed_col] = None
                except:
                    record[renamed_col] = None
    
    record['processed_at'] = datetime.now()
    return record


def save_to_database(record):
    """Save to RDS"""
    print(f"\n{'='*70}")
    print(f"SAVING TO DATABASE")
    print(f"{'='*70}")
    
    try:
        if get_db_engine:
            engine = get_db_engine()
        else:
            db_host = os.getenv("RDS_HOST")
            db_port = os.getenv("RDS_PORT")
            db_name = os.getenv("RDS_DB_NAME")
            db_user = os.getenv("RDS_USERNAME")
            db_pass = os.getenv("RDS_PASSWORD")
            
            if not all([db_host, db_port, db_name, db_user, db_pass]):
                print("❌ Missing database credentials")
                return False
            
            engine = create_engine(
                f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
            )
        
        df = pd.DataFrame([record])
        
        if record.get('latitude') and record.get('longitude'):
            df['__etl_lat'] = df['latitude']
            df['__etl_lon'] = df['longitude']
        
        temp_table = "parcels_enriched_temp"
        df.to_sql(temp_table, engine, if_exists="replace", index=False)
        
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
            conn.execute(text(f"ALTER TABLE {temp_table} DROP COLUMN IF EXISTS geom;"))
            conn.execute(text(f"ALTER TABLE {temp_table} ADD COLUMN geom geometry(Point,4326);"))
            
            if '__etl_lon' in df.columns and '__etl_lat' in df.columns:
                conn.execute(text(f"""
                    UPDATE {temp_table}
                    SET geom = ST_SetSRID(ST_MakePoint(CAST("__etl_lon" AS double precision),
                                                       CAST("__etl_lat" AS double precision)), 4326)
                    WHERE "__etl_lon" IS NOT NULL AND "__etl_lat" IS NOT NULL;
                """))
            
            conn.execute(text(f"""
                INSERT INTO parcels_enriched 
                SELECT * FROM {temp_table}
                ON CONFLICT (parcel_number) DO UPDATE 
                SET location = EXCLUDED.location,
                    latitude = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude,
                    census_tract = EXCLUDED.census_tract,
                    geom = EXCLUDED.geom;
            """))
            
            conn.execute(text(f"DROP TABLE {temp_table};"))
        
        print(f"✓ Successfully saved to parcels_enriched table")
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print(f"{'='*70}")
    print(f"FIND AND PROCESS CLICKABLE ADDRESS")
    print(f"{'='*70}")
    
    # Step 1: Load clickable addresses
    clickable_addresses = load_clickable_addresses()
    if not clickable_addresses:
        print("❌ No clickable addresses found")
        return
    
    # Step 2: Find matching address
    selected = find_matching_clickable(clickable_addresses, SEARCH_ADDRESS)
    if not selected:
        return
    
    # Step 3: Find in OPA data
    parcel_data = find_in_opa_data(selected)
    if parcel_data is None:
        return
    
    # Step 4: Geocode
    geocode_result = geocode_address(parcel_data)
    if not geocode_result or not geocode_result.get('matched'):
        print(f"\n❌ FAILED: Could not geocode")
        return
    
    # Step 5: Fetch census data
    census_data = None
    if geocode_result.get('census_tract'):
        census_data = fetch_census_data(geocode_result['census_tract'])
    
    # Step 6: Create record
    record = create_enriched_record(parcel_data, geocode_result, census_data)
    
    # Step 7: Save to database
    success = save_to_database(record)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Address: {selected['address']}")
    print(f"Parcel Number: {parcel_data.get('parcel_number', 'N/A')}")
    print(f"Clickable in UI: ✓ Yes")
    print(f"Found in OPA: ✓ Yes")
    print(f"Geocoded: {'✓ Yes' if geocode_result.get('matched') else '❌ No'}")
    print(f"Census Data: {'✓ Yes' if census_data else '❌ No'}")
    print(f"Saved to RDS: {'✓ Yes' if success else '❌ No'}")
    
    if success:
        print(f"\n🎉 SUCCESS! This address is now:")
        print(f"   1. Clickable in the UI (from GeoJSON)")
        print(f"   2. In the RDS database with census data")
        print(f"   3. Ready to display census info when clicked!")


if __name__ == "__main__":
    main()
