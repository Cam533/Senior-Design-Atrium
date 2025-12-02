# drop databases from the RDS instance

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

def drop_database(database_name, confirm=False):
    """Drop a database from the RDS instance"""
    # Connect to postgres database to drop other databases
    conn = get_rds_connection('postgres')
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    try:
        # Check if database exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM pg_database 
                WHERE datname = %s
            );
        """, (database_name,))
        exists = cur.fetchone()[0]
        
        if not exists:
            print(f"Database '{database_name}' does not exist.")
            cur.close()
            conn.close()
            return False
        
        # Prevent dropping critical databases
        if database_name in ['postgres', 'rdsadmin', 'template0', 'template1']:
            print(f"Error: Cannot drop system database '{database_name}'.")
            cur.close()
            conn.close()
            return False
        
        if not confirm:
            response = input(f"Are you sure you want to drop database '{database_name}'? This will delete ALL data! (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Operation cancelled.")
                cur.close()
                conn.close()
                return False
        
        # Drop the database
        # Terminate all connections to the database first
        cur.execute("""
            SELECT pg_terminate_backend(pg_stat_activity.pid)
            FROM pg_stat_activity
            WHERE pg_stat_activity.datname = %s
            AND pid <> pg_backend_pid();
        """, (database_name,))
        
        # Now drop the database
        cur.execute(f'DROP DATABASE IF EXISTS "{database_name}";')
        print(f"Database '{database_name}' has been dropped successfully.")
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error dropping database: {e}")
        cur.close()
        conn.close()
        return False

def show_databases():
    """Show all available databases in the RDS instance"""
    databases = list_databases()
    if databases:
        print("Available databases:")
        for db in databases:
            print(f"  - {db}")
    else:
        print("No databases found in the RDS instance.")
    return databases

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drop a database from the RDS instance')
    parser.add_argument('database_name', type=str, nargs='?', help='The name of the database to drop (optional - if not provided, shows available databases)')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()
    
    if args.database_name:
        drop_database(args.database_name, confirm=args.yes)
    else:
        show_databases()

