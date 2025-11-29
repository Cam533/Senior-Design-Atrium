from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text
import sys
import os
import json
import geopandas as gpd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path so we can import models and access
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from access.db_access import get_db_engine # Import your DB helper
from models.rag.query_rag import get_rag_response

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    plotInfo: dict

class NearbyRequest(BaseModel):
    lat: float
    lon: float
    radius_m: int = 100


@app.get("/")
async def root():
    return {"message": "Atrium API is running", "status": "ok"}

@app.post("/chat")
async def chat(request: ChatRequest):
    # Call the RAG system
    response_text = get_rag_response(message=request.message, plotInfo=request.plotInfo)
    return {"message": response_text}

@app.post("/census_nearby")
def census_nearby(req: NearbyRequest):
    # might have to fix this based on etl cols 
    sql = text("""
    SELECT
      census_tract,
      count(*) AS parcel_count,
      avg(COALESCE(tract_median_income::double precision, NULL)) AS mean_median_income,
      avg(COALESCE(tract_population::double precision, NULL)) AS mean_population
    FROM parcels_enriched
    WHERE geom IS NOT NULL
      AND ST_DWithin(geom::geography, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography, :radius)
    GROUP BY census_tract
    ORDER BY parcel_count DESC;
    """)
    engine = get_db_engine()
    with engine.connect() as conn:
        rows = conn.execute(sql, {"lon": req.lon, "lat": req.lat, "radius": req.radius_m}).mappings().all()
    return {"results": [dict(r) for r in rows]}

@app.get("/map")
async def get_map_data():
    """Fetch vacant land polygons from RDS, fallback to local GeoJSON file"""
    # Try RDS first
    try:
        engine = get_db_engine()
        
        # Try without land_rank filter first (in case column doesn't exist or values are NULL)
        # If we get results, limit to 500 for performance
        sql = """
        SELECT objectid, address, zoningbasedistrict as zoning, geometry 
        FROM vacant_indicators_land
        WHERE geometry IS NOT NULL
        LIMIT 500
        """
        
        # Use geopandas to read PostGIS data
        gdf = gpd.read_postgis(sql, con=engine, geom_col='geometry')
        
        if len(gdf) > 0:
            # Convert to GeoJSON string, then parse back to dict
            geojson_str = gdf.to_json()
            return json.loads(geojson_str)
        else:
            print("No data found in RDS, trying local GeoJSON file...")
            raise Exception("No data in RDS")
        
    except Exception as e:
        print(f"Error fetching map data from RDS: {e}")
        print("Falling back to local GeoJSON file...")
        
        # Fallback to local GeoJSON file
        try:
            # Get path to local GeoJSON file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            geojson_path = os.path.join(project_root, "data", "Vacant_Indicators_Land.geojson")
            
            if os.path.exists(geojson_path):
                # Read GeoJSON file directly as JSON (simpler, no conversion issues)
                with open(geojson_path, 'r', encoding='utf-8') as f:
                    geojson_dict = json.load(f)
                
                # Limit to 500 features for performance
                if 'features' in geojson_dict and len(geojson_dict['features']) > 500:
                    geojson_dict['features'] = geojson_dict['features'][:500]
                
                feature_count = len(geojson_dict.get('features', []))
                print(f"Loaded {feature_count} features from local GeoJSON file")
                return geojson_dict
            else:
                print(f"Local GeoJSON file not found at: {geojson_path}")
                return {"type": "FeatureCollection", "features": []}
                
        except Exception as file_error:
            print(f"Error reading local GeoJSON file: {file_error}")
            import traceback
            traceback.print_exc()
            return {"type": "FeatureCollection", "features": []}
