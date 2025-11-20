# peak at tables in the database

import psycopg2 
from db_access import get_db_connection
import argparse

def print_first_n_rows(table_name, limit):
    conn = get_db_connection()
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
    parser.add_argument('table_name', type=str, help='The name of the table to peek at')
    parser.add_argument('--limit', type=int, default=10, help='The number of rows to print')
    args = parser.parse_args()
    print_first_n_rows(args.table_name, args.limit)