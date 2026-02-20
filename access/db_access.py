import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load .env from project root so RDS_* are found regardless of cwd
_env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_env_path)

def _rds_configured():
    host = os.getenv("RDS_HOST")
    return host and host.strip()


def get_db_connection_for_db(database_name):
    """Create psycopg2 connection to a specific database. Requires RDS_* env vars."""
    if not _rds_configured():
        raise ValueError(
            "RDS database not configured. Set RDS_HOST, RDS_USERNAME, and RDS_PASSWORD in .env."
        )
    return psycopg2.connect(
        host=os.getenv("RDS_HOST"),
        port=os.getenv("RDS_PORT", "5432"),
        database=database_name,
        user=os.getenv("RDS_USERNAME"),
        password=os.getenv("RDS_PASSWORD"),
    )


def get_db_engine(database_name):
    """Create SQLAlchemy engine for GeoPandas to_postgis(). Returns None if RDS_* are not set."""
    if not _rds_configured():
        return None
    return create_engine(
        f"postgresql://{os.getenv('RDS_USERNAME')}:{os.getenv('RDS_PASSWORD')}"
        f"@{os.getenv('RDS_HOST')}:{os.getenv('RDS_PORT', '5432')}"
        f"/{database_name}",
        connect_args={"connect_timeout": 10},
    )