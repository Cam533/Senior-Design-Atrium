import pandas as pd
import requests
import time
import os
from typing import Dict, Optional
import json
from datetime import datetime

CENSUS_API_KEY = "c50562fadf7bcd7d4989a392ccbf9c2333fc74d2" 
OPA_DATA_FILE = "../../data/opa_properties_public.csv"
OUTPUT_FILE = "../../data/philadelphia_parcels_enriched.csv"

SAMPLE_SIZE = 1000  # Change to None for full run

# Census variables to fetch
CENSUS_VARIABLES = [
    "NAME",
    "B01003_001E",  # Total population
    "B19013_001E",  # Median household income
    "B01002_001E",  # Median age
    "B09001_001E",  # Population under 18
    "B09020_001E",  # Population 65+
    "B25024_002E",  # Single-family detached
    "B25024_003E",  # Single-family attached
    "B25024_007E",  # 10-19 unit buildings
    "B25024_008E",  # 20-49 unit buildings
    "B25024_009E",  # 50+ unit buildings
    "B25077_001E",  # Median home value
    "B25064_001E",  # Median gross rent
    "B08301_010E",  # Public transit commuters
    "B11001_002E",  # Family households
    "B11001_009E",  # Single-person households
]

def load_opa_data(filepath: str) -> pd.DataFrame:
    """Load OPA property assessment data and keep only essential columns"""
    print(f"Loading OPA data from {filepath}...")
    
    if SAMPLE_SIZE:
        print(f"⚠️  SAMPLE MODE: Loading only {SAMPLE_SIZE:,} properties for testing")
        df = pd.read_csv(filepath, low_memory=False, nrows=SAMPLE_SIZE)
    else:
        df = pd.read_csv(filepath, low_memory=False)
    
    print(f"Loaded {len(df):,} properties")
    print(f"Original columns: {len(df.columns)}")
    
    # Keep only essential columns before processing
    essential_cols = [
        # Property identifiers (REQUIRED)
        'parcel_number',
        'location',
        'zip_code',
        
        # Address components
        'house_number',
        'street_name',
        'unit',
        'mailing_address_1',
        'mailing_city_state',
        'mailing_zip',
        
        # Ownership
        'owner_1',
        'owner_2',
        
        # Property characteristics
        'zoning',
        'total_area',
        'total_livable_area',
        'building_code_description',
        'category_code_description',
        'number_of_bedrooms',
        'number_of_bathrooms',
        'number_of_rooms',
        'number_stories',
        'year_built',
        'exterior_condition',
        'interior_condition',
        'quality_grade',
        
        # Market data
        'market_value',
        'sale_date',
        'sale_price',
        'taxable_building',
        'taxable_land',
        
        # Location
        'geographic_ward',
    ]
    
    # Keep only columns that exist in the dataframe
    cols_to_keep = [col for col in essential_cols if col in df.columns]
    df_clean = df[cols_to_keep]
    
    print(f"Reduced to {len(df_clean.columns)} essential columns")
    print(f"Kept columns: {df_clean.columns.tolist()}")
    
    return df_clean

def prepare_geocoding_batch(df: pd.DataFrame, batch_size: int = 1000) -> list:
    """
    Prepare addresses for Census batch geocoding
    Format: Unique ID, Street address, City, State, ZIP
    """
    print("\nPreparing addresses for geocoding...")
    
    # Adjust column names based on your OPA file structure
    # Common OPA columns: 'location', 'zip_code', or 'mailing_address_1', 'mailing_zip'
    
    geocode_data = df.copy()
    
    # Create full address string - ADJUST THESE COLUMN NAMES
    geocode_data['full_address'] = (
        geocode_data['location'].fillna('') + ', Philadelphia, PA, ' + 
        geocode_data['zip_code'].astype(str).fillna('')
    )
    
    # Format for Census batch geocoder
    geocode_data['batch_format'] = (
        geocode_data['parcel_number'].astype(str) + ',' +
        geocode_data['location'].fillna('') + ',' +
        'Philadelphia,PA,' +
        geocode_data['zip_code'].astype(str).fillna('')
    )
    
    # Split into batches (Census allows up to 10,000 per batch)
    batches = []
    for i in range(0, len(geocode_data), batch_size):
        batch = geocode_data.iloc[i:i + batch_size]
        batches.append(batch)
    
    print(f"Created {len(batches)} batches of up to {batch_size} addresses")
    
    return batches


def geocode_batch(batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Geocode a batch of addresses using Census Geocoder API
    Returns DataFrame with census tract information
    
    Census Batch Geocoder Output Format (for matches):
    Field 0: ID
    Field 1: Input Address
    Field 2: Match Status ("Match" or "No_Match")
    Field 3: Match Type ("Exact" or "Non_Exact")
    Field 4: Matched Address
    Field 5: Coordinates "lon,lat"
    Field 6: TLID
    Field 7: Side
    Field 8: State FIPS
    Field 9: County FIPS
    Field 10: Tract
    Field 11: Block
    """
    print(f"Geocoding batch of {len(batch_df)} addresses...")
    
    # Create CSV content for batch
    csv_content = "id,street,city,state,zip\n"
    for _, row in batch_df.iterrows():
        # Clean the address - remove extra quotes and commas that might break CSV
        street = str(row['location']).replace('"', '').strip()
        zip_code = str(row['zip_code']).strip()
        csv_content += f'"{row["parcel_number"]}","{street}","Philadelphia","PA","{zip_code}"\n'
    
    # Census batch geocoding endpoint
    url = "https://geocoding.geo.census.gov/geocoder/geographies/addressbatch"
    
    files = {
        'addressFile': ('addresses.csv', csv_content, 'text/csv')
    }
    
    data = {
        'benchmark': 'Public_AR_Current',
        'vintage': 'Current_Current'
    }
    
    try:
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            import csv
            from io import StringIO
            
            results = []
            
            for line in response.text.strip().split('\n'):
                try:
                    reader = csv.reader(StringIO(line))
                    parts = next(reader)
                    
                    if len(parts) < 3:
                        continue
                    
                    parcel_id = parts[0]
                    match_status = parts[2]
                    matched = match_status == 'Match'
                    
                    lat = None
                    lon = None
                    tract = None
                    
                    if matched and len(parts) >= 12:
                        # Field 5: Coordinates as "lon,lat"
                        coords = parts[5]
                        if ',' in coords:
                            try:
                                lon_str, lat_str = coords.split(',')
                                lon = float(lon_str.strip())
                                lat = float(lat_str.strip())
                            except:
                                pass
                        
                        # Field 8: State FIPS, Field 9: County FIPS, Field 10: Tract
                        state_fips = parts[8]
                        county_fips = parts[9]
                        tract_code = parts[10]
                        
                        # Build full 11-digit tract GEOID
                        if state_fips and county_fips and tract_code:
                            # Ensure tract is 6 digits (pad with zeros if needed)
                            tract = f"{state_fips}{county_fips}{tract_code.zfill(6)}"
                    
                    results.append({
                        'parcel_number': parcel_id,
                        'matched': matched,
                        'latitude': lat,
                        'longitude': lon,
                        'census_tract': tract,  # Keep as string
                    })
                    
                except Exception as e:
                    print(f"Error parsing line: {e}")
                    continue
            
            # Convert to DataFrame with explicit dtypes
            df_results = pd.DataFrame(results)
            # Ensure census_tract stays as string
            if 'census_tract' in df_results.columns:
                df_results['census_tract'] = df_results['census_tract'].astype(str)
            
            return df_results
        else:
            print(f"Error: {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Geocoding error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def geocode_all_addresses(df: pd.DataFrame) -> pd.DataFrame:
    """Geocode all addresses in batches with progress saving"""
    print("\n=== GEOCODING ALL ADDRESSES ===")
    
    # Check for existing progress
    progress_file = "../../data/geocoding_progress.csv"
    if os.path.exists(progress_file):
        print(f"Found existing progress file. Loading...")
        existing_results = pd.read_csv(progress_file, dtype={'parcel_number': str, 'census_tract': str})
        completed_parcels = set(existing_results['parcel_number'].astype(str))
        print(f"Already geocoded {len(completed_parcels):,} parcels. Resuming...")
        all_results = [existing_results]
    else:
        completed_parcels = set()
        all_results = []
    
    batches = prepare_geocoding_batch(df, batch_size=1000)
    
    for i, batch in enumerate(batches):
        # Skip already processed batches
        batch_parcel_ids = set(batch['parcel_number'].astype(str))
        if batch_parcel_ids.issubset(completed_parcels):
            print(f"Batch {i+1}/{len(batches)} already processed, skipping...")
            continue
        
        print(f"\nProcessing batch {i+1}/{len(batches)}...")
        
        results = geocode_batch(batch)
        if len(results) > 0:
            all_results.append(results)
            
            # Save progress after each batch
            progress_df = pd.concat(all_results, ignore_index=True)
            # Ensure census_tract is saved as string
            progress_df['census_tract'] = progress_df['census_tract'].astype(str)
            progress_df.to_csv(progress_file, index=False)
            print(f"Progress saved: {len(progress_df):,} total geocoded")
        
        # Be nice to the API
        if i < len(batches) - 1:
            time.sleep(2)
    
    if all_results:
        geocoded_df = pd.concat(all_results, ignore_index=True)
        # Final conversion to string
        geocoded_df['census_tract'] = geocoded_df['census_tract'].astype(str)
        print(f"\nGeocoded {len(geocoded_df):,} addresses")
        print(f"Successfully matched: {geocoded_df['matched'].sum():,}")
        
        return geocoded_df
    else:
        return pd.DataFrame()

def get_unique_tracts(geocoded_df: pd.DataFrame) -> list:
    """Get list of unique census tracts"""
    # Ensure census_tract is string type
    geocoded_df['census_tract'] = geocoded_df['census_tract'].astype(str)
    # Get unique tracts, excluding 'nan' and 'None'
    tracts = geocoded_df[geocoded_df['census_tract'].notna()]['census_tract'].unique()
    tracts = [t for t in tracts if t not in ['nan', 'None', '']]
    print(f"\nFound {len(tracts)} unique census tracts")
    print(f"Sample tracts: {tracts[:5]}")
    return tracts


def fetch_census_data_for_tract(tract_geoid: str) -> Optional[Dict]:
    """
    Fetch census data for a specific tract
    Tract GEOID format: SSCCCTTTTTT (state, county, tract)
    """
    # Ensure tract_geoid is a string
    tract_geoid = str(tract_geoid).replace('.0', '')
    
    # Parse GEOID
    if len(tract_geoid) < 11:
        print(f"Invalid tract GEOID (too short): {tract_geoid}")
        return None
    
    state = tract_geoid[:2]
    county = tract_geoid[2:5]
    tract = tract_geoid[5:11]
    
    url = "https://api.census.gov/data/2022/acs/acs5"
    
    params = {
        "get": ",".join(CENSUS_VARIABLES),
        "for": f"tract:{tract}",
        "in": f"state:{state} county:{county}",
        "key": CENSUS_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if len(data) > 1:  # Has header + data row
                header = data[0]
                values = data[1]
                return dict(zip(header, values))
        
        return None
        
    except Exception as e:
        print(f"Error fetching census data for tract {tract_geoid}: {e}")
        return None


def fetch_all_census_data(tracts: list) -> Dict:
    """Fetch census data for all tracts"""
    print("\n=== FETCHING CENSUS DATA ===")
    
    census_cache = {}
    
    for i, tract in enumerate(tracts):
        print(f"Fetching census data for tract {i+1}/{len(tracts)}: {tract}")
        
        data = fetch_census_data_for_tract(tract)
        if data:
            census_cache[tract] = data
        
        # Be nice to the API
        if i < len(tracts) - 1:
            time.sleep(0.5)
    
    print(f"\nSuccessfully fetched data for {len(census_cache)} tracts")
    
    return census_cache


def create_census_dataframe(census_cache: Dict) -> pd.DataFrame:
    """Convert census cache to DataFrame with clean column names"""
    
    if not census_cache:
        print("WARNING: No census data available")
        return pd.DataFrame()
    
    # Convert to DataFrame
    census_df = pd.DataFrame.from_dict(census_cache, orient='index')
    
    # Debug: Check what we got
    print(f"Census data columns before renaming: {census_df.columns.tolist()}")
    
    # Rename columns to be more readable
    column_mapping = {
        "B01003_001E": "tract_total_pop",
        "B19013_001E": "tract_median_income",
        "B01002_001E": "tract_median_age",
        "B09001_001E": "tract_pop_under_18",
        "B09020_001E": "tract_pop_65_plus",
        "B25024_002E": "tract_single_family_detached",
        "B25024_003E": "tract_single_family_attached",
        "B25024_007E": "tract_units_10_19",
        "B25024_008E": "tract_units_20_49",
        "B25024_009E": "tract_units_50_plus",
        "B25077_001E": "tract_median_home_value",
        "B25064_001E": "tract_median_rent",
        "B08301_010E": "tract_transit_commuters",
        "B11001_002E": "tract_family_households",
        "B11001_009E": "tract_single_person_households",
    }
    
    census_df = census_df.rename(columns=column_mapping)
    
    # Keep only the renamed columns
    cols_to_keep = [col for col in column_mapping.values() if col in census_df.columns]
    census_df = census_df[cols_to_keep]
    
    # Convert to numeric
    for col in cols_to_keep:
        census_df[col] = pd.to_numeric(census_df[col], errors='coerce')
    
    # IMPORTANT: Reset index and rename it to census_tract
    census_df.reset_index(inplace=True)
    census_df.rename(columns={'index': 'census_tract'}, inplace=True)
    
    print(f"Census dataframe columns after processing: {census_df.columns.tolist()}")
    print(f"Sample census data:")
    print(census_df.head())
    
    return census_df

def merge_all_data(opa_df: pd.DataFrame, geocoded_df: pd.DataFrame, 
                   census_df: pd.DataFrame) -> pd.DataFrame:
    """Merge OPA, geocoding, and census data"""
    
    print("\n=== MERGING ALL DATA ===")
    
    # Debug: Check what columns we have
    print(f"OPA columns with 'tract': {[c for c in opa_df.columns if 'tract' in c.lower()]}")
    print(f"Geocoded columns: {geocoded_df.columns.tolist()}")
    
    # Ensure parcel_number is the same type in both dataframes
    opa_df['parcel_number'] = opa_df['parcel_number'].astype(str)
    geocoded_df['parcel_number'] = geocoded_df['parcel_number'].astype(str)
    
    # If OPA already has census_tract column, rename it first
    if 'census_tract' in opa_df.columns:
        print("OPA data already has census_tract column, renaming to census_tract_opa")
        opa_df = opa_df.rename(columns={'census_tract': 'census_tract_opa'})
    
    # Check if census_tract column exists in geocoded_df
    if 'census_tract' not in geocoded_df.columns:
        print("WARNING: census_tract column missing from geocoded data!")
        geocoded_df['census_tract'] = None
    
    # Ensure census_tract is string type
    geocoded_df['census_tract'] = geocoded_df['census_tract'].astype(str)
    
    # Select columns to merge
    merge_cols = ['parcel_number', 'matched']
    if 'latitude' in geocoded_df.columns:
        merge_cols.append('latitude')
    if 'longitude' in geocoded_df.columns:
        merge_cols.append('longitude')
    if 'census_tract' in geocoded_df.columns:
        merge_cols.append('census_tract')
    
    # Merge OPA with geocoding results
    merged = opa_df.merge(
        geocoded_df[merge_cols],
        on='parcel_number',
        how='left'
    )
    
    print(f"After geocoding merge: {len(merged):,} rows")
    print(f"Merged columns with 'tract': {[c for c in merged.columns if 'tract' in c.lower()]}")
    
    # Check if census_tract exists after merge
    if 'census_tract' in merged.columns:
        print(f"Properties with census tracts: {merged['census_tract'].notna().sum():,}")
        print(f"Sample census tracts: {merged[merged['census_tract'].notna()]['census_tract'].head().tolist()}")
        
        # Only merge census data if we have census tracts
        if merged['census_tract'].notna().sum() > 0 and len(census_df) > 0:
            print(f"Census dataframe has {len(census_df)} tracts")
            print(f"Census columns: {census_df.columns.tolist()}")
            
            # Merge with census data
            merged = merged.merge(
                census_df,
                on='census_tract',
                how='left'
            )
            print(f"After census merge: {len(merged):,} rows")
            print(f"Columns after census merge: {merged.columns.tolist()[:20]}...")  # Show first 20
        else:
            print("WARNING: No census tracts available or empty census_df, skipping census data merge")
    else:
        print("WARNING: census_tract column not present after merge")
    
    # Add timestamp
    merged['processed_at'] = datetime.now()
    
    print(f"Final dataframe has {len(merged.columns)} columns")
    
    return merged

def main():
    """Run the complete ETL pipeline"""
    
    print("=" * 70)
    print("PHILADELPHIA PROPERTY DATA ETL PIPELINE")
    print("=" * 70)
    
    if SAMPLE_SIZE:
        print(f"\n⚠️  RUNNING IN SAMPLE MODE: Processing {SAMPLE_SIZE:,} properties")
        print("Set SAMPLE_SIZE = None in config to process all properties\n")
    
    # Step 1: Load OPA data
    opa_df = load_opa_data(OPA_DATA_FILE)
    
    # Step 2: Geocode all addresses
    geocoded_df = geocode_all_addresses(opa_df)
    
    if len(geocoded_df) == 0:
        print("ERROR: Geocoding failed. Exiting.")
        return
    
    # Step 3: Get unique tracts and fetch census data
    tracts = get_unique_tracts(geocoded_df)
    census_cache = fetch_all_census_data(tracts)
    census_df = create_census_dataframe(census_cache)
    
    # Step 4: Merge everything (no vacant lot data)
    final_df = merge_all_data(opa_df, geocoded_df, census_df)
    
    # Step 5: Save output
    print(f"\n=== SAVING OUTPUT ===")
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved enriched data to {OUTPUT_FILE}")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total properties: {len(final_df):,}")
    print(f"Successfully geocoded: {final_df['matched'].sum():,}")
    print(f"Match rate: {final_df['matched'].sum() / len(final_df) * 100:.1f}%")
    
    # Check if census data was merged
    if 'tract_median_income' in final_df.columns:
        print(f"Properties with census data: {final_df['tract_median_income'].notna().sum():,}")
    else:
        print("Census data not merged (columns missing)")
    
    print("\n=== SAMPLE DATA ===")
    # Show available columns
    display_cols = ['parcel_number', 'location']
    if 'census_tract' in final_df.columns:
        display_cols.append('census_tract')
    if 'tract_median_income' in final_df.columns:
        display_cols.append('tract_median_income')
    if 'tract_median_age' in final_df.columns:
        display_cols.append('tract_median_age')
    if 'matched' in final_df.columns:
        display_cols.append('matched')
    
    print(final_df[display_cols].head(10))
    
    if SAMPLE_SIZE:
        print(f"\n⚠️  This was a SAMPLE run with {SAMPLE_SIZE:,} properties")
        print("To process all properties, set SAMPLE_SIZE = None and run again")


if __name__ == "__main__":
    main()