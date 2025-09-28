import psycopg2
from psycopg2 import sql
from config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_DB
)

conn = psycopg2.connect(
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    dbname=POSTGRES_DB
)

def insert_order_to_db(data_insert_dict):
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB
    )
    cursor = conn.cursor()

    query = sql.SQL("""
        INSERT INTO orders (customer_name, phone, address, book_id, quantity)
        VALUES (%s, %s, %s, %s, %s)
    """)

    values = (
        data_insert_dict["customer_name"],
        data_insert_dict["phone"],
        data_insert_dict["address"],
        data_insert_dict["book_id"],
        data_insert_dict["quantity"]
    )

    cursor.execute(query, values)
    conn.commit()

    cursor.close()
    conn.close()