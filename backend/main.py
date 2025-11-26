from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
import sys
import os
import json
import geopandas as gpd

# Add parent directory to path so we can import models and access
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from access.db_access import get_db_engine # Import your DB helper
from models.rag.query_rag import get_rag_response

app = FastAPI()
DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL)

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
    response_text = get_rag_response(request.message)
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
    with engine.connect() as conn:
        rows = conn.execute(sql, {"lon": req.lon, "lat": req.lat, "radius": req.radius_m}).mappings().all()
    return {"results": [dict(r) for r in rows]}

@app.get("/map")
async def get_map_data():
    """Fetch vacant land polygons from RDS"""
    try:
        engine = get_db_engine()
        
        # Query only what we need (limit 500 for performance)
        # We convert geometry to GeoJSON format
        sql = """
        SELECT objectid, address, zoningbasedistrict as zoning, geometry 
        FROM vacant_indicators_land WHERE land_rank > 0.5
        LIMIT 500
        """
        
        # Use geopandas to read PostGIS data
        gdf = gpd.read_postgis(sql, con=engine, geom_col='geometry')
        
        # Convert to GeoJSON string, then parse back to dict
        # so FastAPI sends it as proper JSON object, not a string
        geojson_str = gdf.to_json()
        return json.loads(geojson_str)
        
    except Exception as e:
        print(f"Error fetching map data: {e}")
        return {"type": "FeatureCollection", "features": []}
