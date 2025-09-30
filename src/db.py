from langchain_community.utilities import SQLDatabase
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
import psycopg2
from psycopg2 import sql
from llm import google_text_embedding_model
from config import (
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME
)

db_uri = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
psql_db = SQLDatabase.from_uri(db_uri)

client = QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=QDRANT_COLLECTION_NAME,
    embedding=google_text_embedding_model
)

db_conn = psycopg2.connect(
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    dbname=POSTGRES_DB
)

def get_qdrant_retriever(k):
    qdrant_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return qdrant_retriever

def check_order_validity(data_check):
    book_id, quantity = data_check["book_id"], data_check["quantity"]

    cursor = db_conn.cursor()

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
    cursor = db_conn.cursor()
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
    db_conn.commit()

    cursor.close()