from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
import sys
import os
import json
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Optional
from fastapi import HTTPException


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
# Database name from env (e.g. RDS_DB_NAME); required by get_db_engine()
#_database_name = os.getenv("RDS_DB_NAME", "postgres")
_database_name = "atrium_census"
engine = get_db_engine(_database_name)

# Cache for combined GeoJSON so the map can be served quickly without
# reading/parsing files on every request.
cached_map_geojson = {"type": "FeatureCollection", "features": []}

def load_vacant_geojson_cache() -> None:
    """Load and combine the vacant indicator GeoJSON files into the
    `cached_map_geojson` global. Safe to call at startup or on demand.
    """
    base = Path(__file__).resolve().parents[1]
    data_dir = base / "data"
    # Prefer scored GeoJSONs when available; fall back to unscored if load fails (e.g. file too large)
    land_scored = data_dir / "Vacant_Indicators_Land_scored.geojson"
    bldg_scored = data_dir / "Vacant_Indicators_Bldg_scored.geojson"
    land = data_dir / "Vacant_Indicators_Land.geojson"
    bldg = data_dir / "Vacant_Indicators_Bldg.geojson"

    pairs = [
        (land_scored if land_scored.exists() else land, land),
        (bldg_scored if bldg_scored.exists() else bldg, bldg),
    ]

    features = []
    for preferred, fallback in pairs:
        for f in (preferred, fallback):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    gj = json.load(fh)
                    if gj.get("type") == "FeatureCollection":
                        feats = gj.get("features", [])
                        features.extend(feats)
                    elif gj.get("type") == "Feature":
                        features.append(gj)
                break  # loaded this layer successfully
            except Exception as e:
                if f == fallback:
                    print(f"Warning: could not load GeoJSON for layer {f.name}: {e}")
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

class AWSUserRequest(BaseModel):
    id: str
    email: str
    user_type: str
    organization: Optional[str] = None
    neighborhood: Optional[str] = None
    other_specify: Optional[str] = None
    created_at: str

class ProjectRequest(BaseModel):
    id: str
    owner_id: str
    name: str
    description: str
    created_at: str


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
    """Return census aggregates near a point. Returns empty results if parcels_enriched is missing."""
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
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"lon": req.lon, "lat": req.lat, "radius": req.radius_m}).mappings().all()
        return {"results": [dict(r) for r in rows]}
    except ProgrammingError as e:
        if "parcels_enriched" in str(e) or "does not exist" in str(e):
            return {"results": []}
        raise


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

@app.post("/add-aws-user")
def add_aws_user(req: AWSUserRequest):
    """Write user profile to AWS (RDS). Request is logged so you can confirm the backend received it."""
    print("[add-aws-user] request received for", req.email)
    sql = text("""
    INSERT INTO aws_user (id, email, user_type, organization, neighborhood, other_specify, created_at)
    VALUES (:id, :email, :user_type, :organization, :neighborhood, :other_specify, :created_at)
    ON CONFLICT (id) DO UPDATE SET
      email = EXCLUDED.email,
      user_type = EXCLUDED.user_type,
      organization = EXCLUDED.organization,
      neighborhood = EXCLUDED.neighborhood,
      other_specify = EXCLUDED.other_specify;
    """)
    try:
        # engine.begin() auto-commits if no exception
        with engine.begin() as conn:
            conn.execute(sql, {
                "id": req.id,
                "email": req.email,
                "user_type": req.user_type,
                "organization": req.organization,
                "neighborhood": req.neighborhood,
                "other_specify": req.other_specify,
                "created_at": req.created_at,
            })
        return {"status": "ok", "message": "user profile written to AWS"}
    except ProgrammingError as e:
        # table missing etc
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-project")
def add_project(req: ProjectRequest):
    """Write project to AWS (RDS). Request is logged so you can confirm the backend received it."""
    sql = text("""
    INSERT INTO project (id, owner_id, name, description, created_at)
    VALUES (:id, :owner_id, :name, :description, :created_at)
    """)
    print("add-project request received for", req)
    try:
        with engine.begin() as conn:
            conn.execute(sql, {
                "id": req.id,
                "owner_id": req.owner_id,
                "name": req.name,
                "description": req.description,
                "created_at": req.created_at,
            })
        return {"status": "ok", "message": "project written to AWS"}
    except ProgrammingError as e:
        # table missing etc
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects")
def get_projects(owner_id: str):
    """Get all projects for a given owner_id."""
    sql = text("""
    SELECT * FROM project WHERE owner_id = :owner_id
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"owner_id": owner_id}).mappings().all()
        return {"projects": [dict(r) for r in rows]}
    except ProgrammingError as e:
        if "project" in str(e) or "does not exist" in str(e):
            return {"projects": []}
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/parcel_census_data")
def parcel_census_data(req: NearbyRequest):
    """Get census and property data for a parcel at given coordinates.
    Returns {"data": None} if parcels_enriched table is missing or no parcel found.
    """
    sql = text("""
    SELECT
      parcel_number,
      location AS address,
      category_code_description,
      census_tract,
      tract_total_pop,
      tract_median_income,
      tract_median_age,
      tract_pop_under_18,
      tract_pop_65_plus,
      tract_median_home_value,
      tract_median_rent,
      tract_transit_commuters,
      tract_family_households,
      tract_single_person_households,
      owner_1,
      zoning,
      year_built,
      market_value
    FROM parcels_enriched
    WHERE geom IS NOT NULL
      AND ST_DWithin(geom::geography, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography, :radius)
    ORDER BY ST_Distance(geom::geography, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography)
    LIMIT 1;
    """)
    try:
        with engine.connect() as conn:
            row = conn.execute(sql, {"lon": req.lon, "lat": req.lat, "radius": req.radius_m}).mappings().fetchone()
        if row:
            return {"data": dict(row)}
        return {"data": None}
    except ProgrammingError as e:
        if "parcels_enriched" in str(e) or "does not exist" in str(e):
            return {"data": None}
        raise


