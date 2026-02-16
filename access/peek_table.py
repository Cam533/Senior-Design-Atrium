# peak at tables in the database

import psycopg2 
from db_access import get_db_connection_for_db
import argparse

def print_first_n_rows(database_name, table_name, limit):
    conn = get_db_connection_for_db(database_name)
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table_name} LIMIT {limit};")
    rows = cur.fetchall()
    for row in rows:
        print(row)
    cur.close()
    conn.close()

if __name__ == "__main__":
    # add command line argument for table name
    parser = argparse.ArgumentParser()
    parser.add_argument('--table_name', type=str, help='The name of the table to peek at')
    parser.add_argument('--limit', type=int, default=10, help='The number of rows to print')
    parser.add_argument('--database', type=str, default='atrium_census', help='The name of the database to peek at')
    args = parser.parse_args()
    print_first_n_rows(args.database, args.table_name, args.limit)