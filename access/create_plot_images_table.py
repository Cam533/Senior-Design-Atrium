"""
Migration script to create the plot_images table in RDS.
This table stores the mapping between plot objects (by parcel_number) and their associated image file IDs in S3.

Run this once to set up the table:
  python access/create_plot_images_table.py
"""

from sqlalchemy import text
from db_access import get_db_engine

def create_plot_images_table():
    """Create the plot_images table if it doesn't exist."""
    engine = get_db_engine("atrium_census")
    
    # SQL to create the table
    sql = text("""
    CREATE TABLE IF NOT EXISTS plot_images (
        id SERIAL PRIMARY KEY,
        parcel_number VARCHAR(255) NOT NULL,
        file_ids TEXT[] NOT NULL DEFAULT '{}',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(parcel_number)
    );
    
    -- Create index on parcel_number for fast lookups
    CREATE INDEX IF NOT EXISTS idx_plot_images_parcel_number 
      ON plot_images(parcel_number);
    """)
    
    try:
        with engine.begin() as conn:
            conn.execute(sql)
        print("✓ plot_images table created successfully")
    except Exception as e:
        print(f"✗ Error creating table: {e}")
        raise

if __name__ == "__main__":
    create_plot_images_table()
