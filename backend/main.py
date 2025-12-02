from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text
import sys
import os
import json
from fastapi.responses import JSONResponse
from pathlib import Path

# Add parent directory to path so we can import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.rag.query_rag import get_rag_response
except (ImportError, ValueError, RuntimeError, FileNotFoundError) as e:
    def get_rag_response(message: str) -> str:
        return "RAG system not available. Please build vectorstore first by running: python models/rag/build_vectorstore.py"

from models.geographic_scoring import score_location
from access.db_access import get_db_engine

app = FastAPI()
engine = get_db_engine()

# Cache for combined GeoJSON so the map can be served quickly without
# reading/parsing files on every request.
cached_map_geojson = {"type": "FeatureCollection", "features": []}

def load_vacant_geojson_cache() -> None:
    """Load and combine the vacant indicator GeoJSON files into the
    `cached_map_geojson` global. Safe to call at startup or on demand.
    """
    base = Path(__file__).resolve().parents[1]
    data_dir = base / "data"
    files = [data_dir / "Vacant_Indicators_Land.geojson", data_dir / "Vacant_Indicators_Bldg.geojson"]

    features = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                gj = json.load(fh)
                if gj.get("type") == "FeatureCollection":
                    feats = gj.get("features", [])
                    features.extend(feats)
                elif gj.get("type") == "Feature":
                    features.append(gj)
        except Exception:
            # If a file is missing or invalid, skip it.
            continue

    cached_map_geojson["type"] = "FeatureCollection"
    cached_map_geojson["features"] = features


# Load cache at startup so the first map requests are fast.
@app.on_event("startup")
def startup_load_geojson() -> None:
    try:
        load_vacant_geojson_cache()
    except Exception as e:
        print("Warning: failed to load vacant geojson cache:", e)

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
    # Query uses column names from ETL: tract_total_pop, tract_median_income, tract_median_age
    sql = text("""
    SELECT
      census_tract,
      count(*) AS parcel_count,
      avg(COALESCE(tract_median_income::double precision, NULL)) AS mean_median_income,
      avg(COALESCE(tract_total_pop::double precision, NULL)) AS mean_population,
      avg(COALESCE(tract_median_age::double precision, NULL)) AS mean_median_age
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
def map_geojson():
    """Return combined GeoJSON from data/Vacant_Indicators_Land.geojson and Vacant_Indicators_Bldg.geojson.
    Falls back to an empty FeatureCollection if files are missing or invalid.
    """
    # Return the preloaded cached GeoJSON. This avoids re-reading and parsing
    # large GeoJSON files on every request which caused visible lag.
    return JSONResponse(content=cached_map_geojson)


@app.post("/map/reload")
def reload_map_geojson():
    """Manually reload the cached GeoJSON from disk. Useful during development
    if the files change and you want the cache refreshed without restarting.
    """
    try:
        load_vacant_geojson_cache()
        return {"status": "ok", "message": "map cache reloaded"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/geographic_scores")
def geographic_scores(req: NearbyRequest):
    scores = score_location(req.lat, req.lon)
    return scores
