# show databases in the RDS instance and tables in each database

import psycopg2 
import os
from dotenv import load_dotenv
import argparse

load_dotenv()

def get_rds_connection(database='postgres'):
    """Create psycopg2 connection to RDS PostgreSQL database"""
    return psycopg2.connect(
        host=os.getenv('RDS_HOST'),
        port=os.getenv('RDS_PORT', '5432'),
        database=database,
        user=os.getenv('RDS_USERNAME'),
        password=os.getenv('RDS_PASSWORD')
    )

def list_databases():
    """List all databases in the RDS instance"""
    conn = get_rds_connection('postgres')
    cur = conn.cursor()
    
    try:
        # Get all databases (excluding system databases)
        cur.execute("""
            SELECT datname 
            FROM pg_database 
            WHERE datistemplate = false
            AND datname NOT IN ('postgres', 'rdsadmin')
            ORDER BY datname;
        """)
        databases = [db[0] for db in cur.fetchall()]
        
        # Also include 'postgres' if it exists
        cur.execute("""
            SELECT datname 
            FROM pg_database 
            WHERE datname = 'postgres';
        """)
        postgres_db = cur.fetchone()
        if postgres_db:
            databases.insert(0, 'postgres')
        
        cur.close()
        conn.close()
        return databases
        
    except Exception as e:
        print(f"Error listing databases: {e}")
        cur.close()
        conn.close()
        return None

def list_tables_in_database(database_name, show_details=False):
    """List all tables in a specific database"""
    conn = get_rds_connection(database_name)
    cur = conn.cursor()
    
    try:
        if show_details:
            # Get table names with column counts and sizes
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
                print(f"\n  Tables in '{database_name}':")
                print(f"  {'Table Name':<30} {'Columns':<10} {'Size':<15}")
                print(f"  {'-' * 55}")
                for table_name, column_count, size in tables:
                    print(f"  {table_name:<30} {column_count:<10} {size:<15}")
            else:
                print(f"\n  No tables found in database '{database_name}'.")
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
                print(f"\n  Tables in '{database_name}':")
                for table in tables:
                    print(f"    - {table}")
            else:
                print(f"\n  No tables found in database '{database_name}'.")
        
        cur.close()
        conn.close()
        return tables
        
    except Exception as e:
        print(f"  Error listing tables in '{database_name}': {e}")
        cur.close()
        conn.close()
        return None

def show_rds_structure(database_name=None, show_details=False):
    """Show databases and their tables in the RDS instance"""
    print("RDS Instance Structure")
    print("=" * 60)
    
    # List all databases
    databases = list_databases()
    
    if not databases:
        print("No databases found or error connecting to RDS instance.")
        return
    
    print(f"\nDatabases in RDS instance ({len(databases)} total):")
    for db in databases:
        print(f"  - {db}")
    
    # Show tables for specified database or all databases
    if database_name:
        if database_name in databases:
            list_tables_in_database(database_name, show_details)
        else:
            print(f"\nError: Database '{database_name}' not found.")
            print(f"Available databases: {', '.join(databases)}")
    else:
        # Show tables for all databases
        for db in databases:
            list_tables_in_database(db, show_details)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show databases in RDS instance and their tables')
    parser.add_argument('--database', '-db', type=str, help='Show tables for a specific database only')
    parser.add_argument('--details', '-d', action='store_true', help='Show detailed information (column count, size)')
    args = parser.parse_args()
    
    show_rds_structure(database_name=args.database, show_details=args.details)

