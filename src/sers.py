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

def check_order_validity(data_check):
    book_id, quantity = data_check["book_id"], data_check["quantity"]

    cursor = conn.cursor()

    cursor.execute("SELECT stock FROM books WHERE book_id = %s", (book_id,))

    result = cursor.fetchone()
    if result is None:
        cursor.close()
        raise ValueError(f"Book ID '{book_id}' does not exist in the inventory.")
    
    stock = result[0]

    if quantity > stock:
        cursor.close()
        raise ValueError(f"Quantity ({quantity}) exceeds stock ({stock}) for book ID '{book_id}'.")    
    
    cursor.close()

def insert_order_to_db(data_insert):
    cursor = conn.cursor()
    query = sql.SQL("""
        INSERT INTO orders (customer_name, phone, address, book_id, quantity)
        VALUES (%s, %s, %s, %s, %s)
    """)

    values = (
        data_insert["customer_name"],
        data_insert["phone"],
        data_insert["address"],
        data_insert["book_id"],
        data_insert["quantity"]
    )

    cursor.execute(query, values)
    conn.commit()

    cursor.close()