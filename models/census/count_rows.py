import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# path stuff idk 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

print("DEBUG RDS_HOST:", os.getenv("RDS_HOST"))

try:
    from access.db_access import get_db_engine
    engine = get_db_engine()
except Exception as e:
    print("Warning: Could not import get_db_engine, using fallback engine.")
    print("Import error:", e)

    db_host = os.getenv("RDS_HOST")
    db_port = os.getenv("RDS_PORT")
    db_name = os.getenv("RDS_DB_NAME")
    db_user = os.getenv("RDS_USERNAME")
    db_pass = os.getenv("RDS_PASSWORD")

    engine = create_engine(
        f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    )

with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM parcels_enriched;"))
    print("Row count:", result.scalar())
