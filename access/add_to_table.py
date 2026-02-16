import pandas as pd
import geopandas as gpd
from db_access import get_db_connection

def add_to_aws_user_table(id, email, user_type, organization, neighborhood, other_specify, conn):
    """Add to AWS user table"""
    add_to_table_sql = f'''
    INSERT INTO aws_user (id, email, user_type, organization, neighborhood, other_specify)
    VALUES (%s, %s, %s, %s, %s, %s);
    '''
    with conn.cursor() as cur:
        cur.execute(add_to_table_sql, (id, email, user_type, organization, neighborhood, other_specify))
        conn.commit()
        print(f"Added to AWS user table")
    conn.close()