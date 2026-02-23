from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
import sys
import os
import json
import requests
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Optional, List
from fastapi import HTTPException
import boto3
from botocore.exceptions import ClientError
import uuid
import io
import mimetypes

from dotenv import load_dotenv

# Load .env from project root (parent of backend/)
_env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_env_path)

# Add parent directory to path so we can import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize S3 client for image uploads
_s3_client = None
_s3_bucket_name = os.getenv("S3_BUCKET_NAME")

def get_s3_client():
    """Get or create the S3 client."""
    global _s3_client
    if _s3_client is None:
        try:
            _s3_client = boto3.client(
                "s3",
                region_name=os.getenv("AWS_REGION", "us-east-1"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
        except Exception as e:
            print(f"Warning: Could not initialize S3 client: {e}")
            _s3_client = False  # Mark as failed so we don't retry
    return _s3_client if _s3_client else None

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
engine = get_db_engine(_database_name)  # None if RDS_* not set in .env

_RDS_REQUIRED_MSG = "RDS database not configured. Set RDS_HOST, RDS_USERNAME, and RDS_PASSWORD in .env for map/parcel/census features."

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


def _feature_center(feature):
    """Extract (lat, lon) from a GeoJSON feature's geometry. Returns (None, None) if not available."""
    try:
        geom = feature.get("geometry")
        if not geom or not geom.get("coordinates"):
            return None, None
        coords = geom["coordinates"]
        if geom.get("type") == "Polygon" and coords and coords[0]:
            ring = coords[0]
            n = len(ring)
            if n == 0:
                return None, None
            sum_lon = sum(c[0] for c in ring)
            sum_lat = sum(c[1] for c in ring)
            return sum_lat / n, sum_lon / n
        if geom.get("type") == "MultiPolygon" and coords and coords[0] and coords[0][0]:
            ring = coords[0][0]
            n = len(ring)
            if n == 0:
                return None, None
            sum_lon = sum(c[0] for c in ring)
            sum_lat = sum(c[1] for c in ring)
            return sum_lat / n, sum_lon / n
    except Exception:
        pass
    return None, None


class ParcelsByIdsRequest(BaseModel):
    objectids: Optional[List[str]] = []


# Load cache at startup so the first map requests are fast.
@app.on_event("startup")
def startup_load_geojson() -> None:
    try:
        load_vacant_geojson_cache()
    except Exception as e:
        print("Warning: failed to load vacant geojson cache:", e)

    # Ensure liked_lots table exists when DB is available
    if engine is not None:
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS liked_lots (
                    user_id TEXT NOT NULL,
                    parcel_key TEXT NOT NULL,
                    parcel JSONB NOT NULL,
                    liked_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (user_id, parcel_key)
                );
                """))
        except Exception as e:
            print("Warning: failed to ensure liked_lots table:", e)

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
    description: Optional[str] = ""
    plots: Optional[List[str]] = []
    created_at: str

class PlotImagesResponse(BaseModel):
    """Response containing file IDs for a plot."""
    parcel_number: str
    file_ids: List[str]
    image_urls: List[str]

class LikedLotToggleRequest(BaseModel):
    user_id: str
    parcel_key: str
    parcel: dict

class LikedLotStatusResponse(BaseModel):
    liked: bool
    total_likes: int


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
    if engine is None:
        raise HTTPException(status_code=503, detail=_RDS_REQUIRED_MSG)
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


@app.post("/parcels_by_ids")
def parcels_by_ids(req: ParcelsByIdsRequest):
    """Return parcel objects (with lat, lon) for the given objectids from the cached map GeoJSON."""
    want = {str(oid).strip() for oid in (req.objectids or []) if oid is not None and str(oid).strip()}
    if not want:
        return {"parcels": []}
    parcels = []
    for f in cached_map_geojson.get("features") or []:
        props = f.get("properties") or {}
        oid = props.get("objectid")
        if oid is None:
            continue
        if str(oid).strip() not in want:
            continue
        lat, lon = _feature_center(f)
        parcel = dict(props)
        if lat is not None:
            parcel["lat"] = lat
        if lon is not None:
            parcel["lon"] = lon
        parcels.append(parcel)
    return {"parcels": parcels}


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

@app.get("/liked_lots")
def list_liked_lots(user_id: str):
    if engine is None:
        raise HTTPException(status_code=503, detail=_RDS_REQUIRED_MSG)
    sql = text("""
    SELECT parcel_key, parcel, liked_at
    FROM liked_lots
    WHERE user_id = :user_id
    ORDER BY liked_at DESC;
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"user_id": user_id}).mappings().all()
    return {"items": [dict(r) for r in rows]}

@app.get("/liked_lots/count")
def liked_lots_count(parcel_key: str):
    if engine is None:
        raise HTTPException(status_code=503, detail=_RDS_REQUIRED_MSG)
    sql = text("SELECT COUNT(*) AS total FROM liked_lots WHERE parcel_key = :parcel_key")
    with engine.connect() as conn:
        row = conn.execute(sql, {"parcel_key": parcel_key}).mappings().first()
    return {"total": int(row["total"]) if row else 0}

@app.get("/liked_lots/status")
def liked_lot_status(user_id: str, parcel_key: str):
    if engine is None:
        raise HTTPException(status_code=503, detail=_RDS_REQUIRED_MSG)
    sql = text("""
    SELECT 1
    FROM liked_lots
    WHERE user_id = :user_id AND parcel_key = :parcel_key
    LIMIT 1;
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"user_id": user_id, "parcel_key": parcel_key}).first()
    return {"liked": row is not None}

@app.post("/liked_lots/toggle")
def toggle_liked_lot(req: LikedLotToggleRequest) -> LikedLotStatusResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail=_RDS_REQUIRED_MSG)
    with engine.begin() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM liked_lots WHERE user_id = :user_id AND parcel_key = :parcel_key"),
            {"user_id": req.user_id, "parcel_key": req.parcel_key},
        ).first()

        if exists:
            conn.execute(
                text("DELETE FROM liked_lots WHERE user_id = :user_id AND parcel_key = :parcel_key"),
                {"user_id": req.user_id, "parcel_key": req.parcel_key},
            )
            liked = False
        else:
            conn.execute(
                text("""
                INSERT INTO liked_lots (user_id, parcel_key, parcel)
                VALUES (:user_id, :parcel_key, :parcel)
                ON CONFLICT (user_id, parcel_key)
                DO UPDATE SET parcel = EXCLUDED.parcel, liked_at = NOW();
                """),
                {"user_id": req.user_id, "parcel_key": req.parcel_key, "parcel": json.dumps(req.parcel)},
            )
            liked = True

        total_row = conn.execute(
            text("SELECT COUNT(*) AS total FROM liked_lots WHERE parcel_key = :parcel_key"),
            {"parcel_key": req.parcel_key},
        ).mappings().first()
        total = int(total_row["total"]) if total_row else 0

    return {"liked": liked, "total_likes": total}

@app.post("/add-aws-user")
def add_aws_user(req: AWSUserRequest):
    """Write user profile to AWS (RDS). Request is logged so you can confirm the backend received it."""
    if engine is None:
        raise HTTPException(status_code=503, detail=_RDS_REQUIRED_MSG)
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
    if engine is None:
        raise HTTPException(status_code=503, detail=_RDS_REQUIRED_MSG)
    sql = text("""
    INSERT INTO project (id, owner_id, name, description, plots, created_at)
    VALUES (:id, :owner_id, :name, :description, :plots, :created_at)
    """)
    print("add-project request received for", req)
    try:
        description = req.description if req.description is not None else ""
        plots = req.plots if req.plots is not None else []
        with engine.begin() as conn:
            conn.execute(sql, {
                "id": req.id,
                "owner_id": req.owner_id,
                "name": req.name,
                "description": description,
                "plots": plots,
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
    if engine is None:
        raise HTTPException(status_code=503, detail=_RDS_REQUIRED_MSG)
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
    if engine is None:
        raise HTTPException(status_code=503, detail=_RDS_REQUIRED_MSG)
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


def _supabase_headers(service_role_key: str):
    return {"Authorization": f"Bearer {service_role_key}", "apikey": service_role_key, "Content-Type": "application/json"}


@app.post("/upload-plot-image")
async def upload_plot_image(
    parcel_number: str,
    file: UploadFile = File(...)
):
    """
    Upload an image for a plot/parcel.
    - Stores the image in S3 with a unique file ID
    - Records the file ID in the plot_images table
    - Returns the file ID and S3 URL
    """
    if engine is None:
        raise HTTPException(status_code=503, detail=_RDS_REQUIRED_MSG)
    
    s3_client = get_s3_client()
    if not s3_client or not _s3_bucket_name:
        raise HTTPException(
            status_code=503,
            detail="S3 not configured. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and S3_BUCKET_NAME in .env"
        )
    
    try:
        # Read file content
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Determine file extension
        ext = ""
        if file.filename:
            _, ext = os.path.splitext(file.filename)
        
        # Upload to S3
        s3_key = f"plot-images/{parcel_number}/{file_id}{ext}"
        
        try:
            s3_client.put_object(
                Bucket=_s3_bucket_name,
                Key=s3_key,
                Body=io.BytesIO(content),
                ContentType=file.content_type or "application/octet-stream",
            )
        except ClientError as e:
            print(f"S3 upload error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {str(e)}")
        
        # Update or insert into plot_images table
        sql = text("""
        INSERT INTO plot_images (parcel_number, file_ids)
        VALUES (:parcel_number, ARRAY[:file_id])
        ON CONFLICT (parcel_number) DO UPDATE SET
          file_ids = array_append(plot_images.file_ids, EXCLUDED.file_ids[1]),
          updated_at = CURRENT_TIMESTAMP
        """)
        
        try:
            with engine.begin() as conn:
                conn.execute(sql, {"parcel_number": parcel_number, "file_id": file_id})
        except ProgrammingError as e:
            if "plot_images" in str(e) or "does not exist" in str(e):
                raise HTTPException(
                    status_code=500,
                    detail="plot_images table not found. Run: python access/create_plot_images_table.py"
                )
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
        # Generate S3 URL
        s3_url = f"https://{_s3_bucket_name}.s3.amazonaws.com/{s3_key}"
        
        return {
            "status": "ok",
            "file_id": file_id,
            "s3_url": s3_url,
            "parcel_number": parcel_number
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error uploading plot image: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading image: {str(e)}")


@app.get("/plot-images/{parcel_number}")
def get_plot_images(parcel_number: str):
    """
    Get all images for a plot/parcel.
    Returns file IDs and pre-signed S3 URLs.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail=_RDS_REQUIRED_MSG)
    
    s3_client = get_s3_client()
    
    sql = text("""
    SELECT file_ids FROM plot_images WHERE parcel_number = :parcel_number
    """)
    
    try:
        with engine.connect() as conn:
            row = conn.execute(sql, {"parcel_number": parcel_number}).mappings().fetchone()
    except ProgrammingError as e:
        if "plot_images" in str(e) or "does not exist" in str(e):
            return {"parcel_number": parcel_number, "file_ids": [], "image_urls": []}
        raise HTTPException(status_code=500, detail=str(e))
    
    if not row or not row.file_ids:
        return {"parcel_number": parcel_number, "file_ids": [], "image_urls": []}
    
    file_ids = list(row.file_ids) if row.file_ids else []
    image_urls = []
    
    # Generate S3 URLs for each file
    if s3_client and _s3_bucket_name:
        for file_id in file_ids:
            # Try to find the file with common extensions
            found = False
            for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ""]:
                s3_key = f"plot-images/{parcel_number}/{file_id}{ext}"
                try:
                    # Check if object exists and generate URL
                    s3_client.head_object(Bucket=_s3_bucket_name, Key=s3_key)
                    url = f"https://{_s3_bucket_name}.s3.amazonaws.com/{s3_key}"
                    image_urls.append(url)
                    found = True
                    break
                except ClientError:
                    continue
            
            # Fallback to generic URL if we couldn't find the exact extension
            if not found:
                url = f"https://{_s3_bucket_name}.s3.amazonaws.com/plot-images/{parcel_number}/{file_id}"
                image_urls.append(url)
    else:
        # If no S3 client, just return placeholder URLs
        for file_id in file_ids:
            image_urls.append(f"s3://plot-images/{parcel_number}/{file_id}")
    
    return {"parcel_number": parcel_number, "file_ids": file_ids, "image_urls": image_urls}


@app.delete("/plot-image/{parcel_number}/{file_id}")
def delete_plot_image(parcel_number: str, file_id: str):
    """
    Delete a specific image from a plot.
    Removes from both S3 and the database.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail=_RDS_REQUIRED_MSG)
    
    s3_client = get_s3_client()
    if not s3_client or not _s3_bucket_name:
        raise HTTPException(
            status_code=503,
            detail="S3 not configured"
        )
    
    # Delete from S3
    for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ""]:
        s3_key = f"plot-images/{parcel_number}/{file_id}{ext}"
        try:
            s3_client.delete_object(Bucket=_s3_bucket_name, Key=s3_key)
            break
        except ClientError:
            continue
    
    # Remove file_id from database
    sql = text("""
    UPDATE plot_images
    SET file_ids = array_remove(file_ids, :file_id),
        updated_at = CURRENT_TIMESTAMP
    WHERE parcel_number = :parcel_number
    """)
    
    try:
        with engine.begin() as conn:
            conn.execute(sql, {"parcel_number": parcel_number, "file_id": file_id})
        return {"status": "ok", "message": "Image deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete-account")
def delete_account(request: Request):
    """Delete the authenticated user's Supabase Auth account and their row in public.users.
    Also removes their avatar from Storage if present. Requires Authorization: Bearer <access_token>."""
    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_role_key:
        raise HTTPException(
            status_code=503,
            detail="Delete account is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.",
        )
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    access_token = auth_header[7:].strip()
    base = supabase_url.rstrip("/")
    get_user_url = f"{base}/auth/v1/user"
    get_headers = {"Authorization": f"Bearer {access_token}", "apikey": service_role_key}
    try:
        r = requests.get(get_user_url, headers=get_headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        user_id = data.get("id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Could not identify user.")
    except requests.RequestException as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    admin_headers = _supabase_headers(service_role_key)

    # 1. Delete user's row in public.users (Supabase Postgres)
    try:
        rest_delete = requests.delete(
            f"{base}/rest/v1/users?id=eq.{user_id}",
            headers={**admin_headers, "Prefer": "return=minimal"},
            timeout=10,
        )
        if rest_delete.status_code not in (200, 204):
            pass  # Row may not exist; continue to auth delete
    except requests.RequestException:
        pass

    # 2. Remove avatar files from Storage (bucket: avatars, keys like <user_id>.*)
    try:
        list_resp = requests.post(
            f"{base}/storage/v1/object/list/avatars",
            headers=admin_headers,
            json={"prefix": user_id, "limit": 20},
            timeout=10,
        )
        if list_resp.status_code == 200:
            for obj in list_resp.json() or []:
                name = obj.get("name")
                if name:
                    requests.delete(f"{base}/storage/v1/object/avatars/{name}", headers=admin_headers, timeout=10)
    except requests.RequestException:
        pass

    # 3. Delete the auth user (required for login to stop working)
    delete_url = f"{base}/auth/v1/admin/users/{user_id}"
    try:
        del_r = requests.delete(delete_url, headers=admin_headers, timeout=10)
        if del_r.status_code == 404:
            return {"status": "ok", "message": "User already deleted."}
        del_r.raise_for_status()
    except requests.RequestException as e:
        resp = getattr(e, "response", None)
        msg = resp.text if resp is not None else str(e)
        raise HTTPException(status_code=502, detail="Failed to delete user: " + msg)
    return {"status": "ok", "message": "Account deleted."}
