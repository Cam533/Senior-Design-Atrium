# list all tables in the database

import psycopg2 
import os
from dotenv import load_dotenv
from db_access import get_db_connection
import argparse

def list_tables(show_details=False):
    """Show all available tables in the database"""
    if database_name:
        conn = get_db_connection_for_db(database_name)
    else:
        conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        if show_details:
            # Get table names with row counts
            cur.execute("""
                SELECT 
                    t.table_name,
                    (SELECT COUNT(*) FROM information_schema.columns 
                     WHERE table_schema = 'public' AND table_name = t.table_name) as column_count,
                    pg_size_pretty(pg_total_relation_size(quote_ident(t.table_name)::regclass)) as size
                FROM information_schema.tables t
                WHERE t.table_schema = 'public' 
                ORDER BY t.table_name;
            """)
            tables = cur.fetchall()
            
            if tables:
                print("Available tables:")
                print(f"{'Table Name':<30} {'Columns':<10} {'Size':<15}")
                print("-" * 55)
                for table_name, column_count, size in tables:
                    print(f"{table_name:<30} {column_count:<10} {size:<15}")
            else:
                print("No tables found in the database.")
        else:
            # Simple list of table names
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name;
            """)
            tables = [table[0] for table in cur.fetchall()]
            
            if tables:
                print("Available tables:")
                for table in tables:
                    print(f"  - {table}")
            else:
                print("No tables found in the database.")
        
        cur.close()
        conn.close()
        return tables
        
    except Exception as e:
        print(f"Error listing tables: {e}")
        cur.close()
        conn.close()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='List all tables in the database')
    parser.add_argument('--database', '-db', type=str, help='The name of the database (optional - uses default from .env if not provided)')
    parser.add_argument('--details', '-d', action='store_true', help='Show detailed information (column count, size)')
    args = parser.parse_args()
    
    list_tables(show_details=args.details, database_name=args.database)

