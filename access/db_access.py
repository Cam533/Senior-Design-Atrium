import psycopg2
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

def get_db_connection():
    """Create psycopg2 connection to RDS PostgreSQL database"""
    return psycopg2.connect(
        host=os.getenv('RDS_HOST'),
        port=os.getenv('RDS_PORT', '5432'),
        database=os.getenv('RDS_DB_NAME', 'postgres'),
        user=os.getenv('RDS_USERNAME'),
        password=os.getenv('RDS_PASSWORD')
    )

def get_db_engine():
    """Create SQLAlchemy engine for GeoPandas to_postgis()"""
    return create_engine(
        f"postgresql://{os.getenv('RDS_USERNAME')}:{os.getenv('RDS_PASSWORD')}"
        f"@{os.getenv('RDS_HOST')}:{os.getenv('RDS_PORT', '5432')}"
        f"/{os.getenv('RDS_DB_NAME', 'postgres')}"
    )