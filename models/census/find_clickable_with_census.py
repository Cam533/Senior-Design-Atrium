"""
Script to find addresses that are both:
1. Clickable in the UI (present in GeoJSON files)
2. Have census data in the RDS database

This helps identify which parcels will display census data when clicked.
"""

import json
import sys
import os
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# path stuff - go up two levels to reach project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

print("DEBUG RDS_HOST:", os.getenv("RDS_HOST"))

try:
    from access.db_access import get_db_engine
    engine = get_db_engine()
    print("✓ Connected using get_db_engine()")
except Exception as e:
    print("Warning: Could not import get_db_engine, using fallback engine.")
    print("Import error:", e)

    db_host = os.getenv("RDS_HOST")
    db_port = os.getenv("RDS_PORT")
    db_name = os.getenv("RDS_DB_NAME")
    db_user = os.getenv("RDS_USERNAME")
    db_pass = os.getenv("RDS_PASSWORD")

    if not all([db_host, db_port, db_name, db_user, db_pass]):
        print("Error: Missing required environment variables!")
        print(f"RDS_HOST: {db_host}")
        print(f"RDS_PORT: {db_port}")
        print(f"RDS_DB_NAME: {db_name}")
        print(f"RDS_USERNAME: {'***' if db_user else 'None'}")
        print(f"RDS_PASSWORD: {'***' if db_pass else 'None'}")
        sys.exit(1)

    engine = create_engine(
        f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    )
    print("✓ Connected using fallback engine")


def load_geojson_addresses():
    """Load all addresses from the GeoJSON files (clickable in UI)"""
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    
    geojson_files = [
        data_dir / "Vacant_Indicators_Land.geojson",
        data_dir / "Vacant_Indicators_Bldg.geojson"
    ]
    
    addresses = {}
    
    for filepath in geojson_files:
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping...")
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if data.get("type") == "FeatureCollection":
                features = data.get("features", [])
                for feature in features:
                    props = feature.get("properties", {})
                    
                    # Get address (might be in 'address' or 'location' field)
                    address = props.get("address") or props.get("location")
                    opa_id = props.get("opa_id") or props.get("parcel_number")
                    objectid = props.get("objectid")
                    
                    # Get coordinates for matching
                    coords = None
                    if feature.get("geometry"):
                        geom = feature["geometry"]
                        if geom.get("type") == "Polygon" and geom.get("coordinates"):
                            # Calculate centroid of polygon
                            polygon = geom["coordinates"][0]
                            if polygon:
                                lon = sum(c[0] for c in polygon) / len(polygon)
                                lat = sum(c[1] for c in polygon) / len(polygon)
                                coords = (lat, lon)
                        elif geom.get("type") == "MultiPolygon" and geom.get("coordinates"):
                            first_polygon = geom["coordinates"][0][0]
                            if first_polygon:
                                lon = sum(c[0] for c in first_polygon) / len(first_polygon)
                                lat = sum(c[1] for c in first_polygon) / len(first_polygon)
                                coords = (lat, lon)
                    
                    if address or opa_id:
                        key = address or opa_id
                        addresses[key] = {
                            "address": address,
                            "opa_id": opa_id,
                            "objectid": objectid,
                            "coords": coords,
                            "source_file": filepath.name
                        }
                        
            print(f"Loaded {len(features)} features from {filepath.name}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    print(f"\nTotal unique addresses in GeoJSON files: {len(addresses)}")
    return addresses


def get_addresses_with_census_data():
    """Get all addresses from RDS that have census data"""
    sql = text("""
    SELECT 
        parcel_number,
        location AS address,
        latitude,
        longitude,
        census_tract,
        category_code_description,
        owner_1,
        tract_median_income,
        tract_total_pop
    FROM parcels_enriched
    WHERE census_tract IS NOT NULL
        AND census_tract != 'nan'
        AND census_tract != 'None'
    ORDER BY location;
    """)
    
    try:
        with engine.connect() as conn:
            results = conn.execute(sql).mappings().all()
        
        print(f"\nFound {len(results)} parcels with census data in RDS")
        return [dict(row) for row in results]
    except Exception as e:
        print(f"Error querying database: {e}")
        return []


def find_matches(geojson_addresses, census_parcels):
    """Find parcels that exist in both GeoJSON and census data"""
    matches = []
    
    # Create lookup dictionaries for faster matching
    geojson_by_address = {addr.lower().strip(): data 
                          for addr, data in geojson_addresses.items() 
                          if data.get("address")}
    geojson_by_opa = {data.get("opa_id"): data 
                      for data in geojson_addresses.values() 
                      if data.get("opa_id")}
    
    for parcel in census_parcels:
        matched = False
        match_type = None
        geojson_data = None
        
        # Try matching by address
        address = parcel.get("address", "").lower().strip()
        if address and address in geojson_by_address:
            matched = True
            match_type = "address"
            geojson_data = geojson_by_address[address]
        
        # Try matching by OPA/parcel number
        if not matched:
            parcel_num = parcel.get("parcel_number")
            if parcel_num and parcel_num in geojson_by_opa:
                matched = True
                match_type = "opa_id"
                geojson_data = geojson_by_opa[parcel_num]
        
        # Try matching by proximity (if coordinates available)
        if not matched and parcel.get("latitude") and parcel.get("longitude"):
            parcel_coords = (parcel["latitude"], parcel["longitude"])
            # Check if any GeoJSON feature is within ~50m
            for data in geojson_addresses.values():
                if data.get("coords"):
                    gjson_coords = data["coords"]
                    # Rough distance check (0.0005 degrees ≈ 50m)
                    lat_diff = abs(gjson_coords[0] - parcel_coords[0])
                    lon_diff = abs(gjson_coords[1] - parcel_coords[1])
                    if lat_diff < 0.0005 and lon_diff < 0.0005:
                        matched = True
                        match_type = "coordinates"
                        geojson_data = data
                        break
        
        if matched:
            matches.append({
                "address": parcel.get("address"),
                "parcel_number": parcel.get("parcel_number"),
                "census_tract": parcel.get("census_tract"),
                "category": parcel.get("category_code_description"),
                "owner": parcel.get("owner_1"),
                "median_income": parcel.get("tract_median_income"),
                "population": parcel.get("tract_total_pop"),
                "lat": parcel.get("latitude"),
                "lon": parcel.get("longitude"),
                "match_type": match_type,
                "geojson_objectid": geojson_data.get("objectid") if geojson_data else None,
                "geojson_file": geojson_data.get("source_file") if geojson_data else None
            })
    
    return matches


def main():
    print("=" * 70)
    print("FINDING CLICKABLE ADDRESSES WITH CENSUS DATA")
    print("=" * 70)
    
    # Load data from both sources
    print("\n[1/3] Loading GeoJSON addresses (clickable in UI)...")
    geojson_addresses = load_geojson_addresses()
    
    print("\n[2/3] Loading census data from RDS...")
    census_parcels = get_addresses_with_census_data()
    
    print("\n[3/3] Finding matches...")
    matches = find_matches(geojson_addresses, census_parcels)
    
    # Report results
    print("\n" + "=" * 70)
    print(f"RESULTS: Found {len(matches)} parcels that are BOTH clickable AND have census data")
    print("=" * 70)
    
    if matches:
        # Save to CSV
        output_file = Path(__file__).parent / "output" / "clickable_with_census.csv"
        output_file.parent.mkdir(exist_ok=True)
        
        import csv
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if matches:
                writer = csv.DictWriter(f, fieldnames=matches[0].keys())
                writer.writeheader()
                writer.writerows(matches)
        
        print(f"\n✓ Saved results to: {output_file}")
        
        # Show sample
        print("\nSample of matched addresses:")
        print("-" * 70)
        for i, match in enumerate(matches[:10]):
            print(f"{i+1}. {match['address']}")
            print(f"   Parcel: {match['parcel_number']} | Tract: {match['census_tract']}")
            print(f"   Category: {match['category']}")
            print(f"   Match type: {match['match_type']}")
            print(f"   Coords: ({match['lat']:.6f}, {match['lon']:.6f})")
            print()
        
        if len(matches) > 10:
            print(f"... and {len(matches) - 10} more")
        
        # Statistics
        print("\n" + "=" * 70)
        print("STATISTICS")
        print("=" * 70)
        print(f"Total in GeoJSON (clickable): {len(geojson_addresses)}")
        print(f"Total with census data: {len(census_parcels)}")
        print(f"Intersection (both): {len(matches)}")
        print(f"Match rate: {len(matches)/len(census_parcels)*100:.1f}% of census data is clickable")
        
        # Match type breakdown
        match_types = {}
        for m in matches:
            mt = m['match_type']
            match_types[mt] = match_types.get(mt, 0) + 1
        
        print("\nMatch type breakdown:")
        for mt, count in match_types.items():
            print(f"  {mt}: {count} ({count/len(matches)*100:.1f}%)")
    else:
        print("\n⚠️  No matches found!")
        print("This means the addresses in your GeoJSON files don't match")
        print("the addresses in your RDS census data.")
        print("\nTroubleshooting:")
        print("1. Check if address formats match between sources")
        print("2. Verify OPA IDs/parcel numbers are consistent")
        print("3. Ensure coordinates are in the same projection")


if __name__ == "__main__":
    main()
