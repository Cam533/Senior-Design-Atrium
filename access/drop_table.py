# drop tables from the database

import psycopg2 
from db_access import get_db_connection
import argparse

def drop_table(table_name, confirm=False):
    """Drop a table from the database"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Check if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            );
        """, (table_name,))
        exists = cur.fetchone()[0]
        
        if not exists:
            print(f"Table '{table_name}' does not exist.")
            cur.close()
            conn.close()
            return False
        
        if not confirm:
            response = input(f"Are you sure you want to drop table '{table_name}'? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Operation cancelled.")
                cur.close()
                conn.close()
                return False
        
        # Drop the table
        cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
        conn.commit()
        print(f"Table '{table_name}' has been dropped successfully.")
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error dropping table: {e}")
        conn.rollback()
        cur.close()
        conn.close()
        return False

def show_tables():
    """Show all available tables in the database"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drop a table from the database')
    parser.add_argument('table_name', type=str, nargs='?', help='The name of the table to drop (optional - if not provided, shows available tables)')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()
    
    if args.table_name:
        drop_table(args.table_name, confirm=args.yes)
    else:
        show_tables()

