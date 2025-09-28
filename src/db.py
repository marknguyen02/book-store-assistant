from langchain_community.utilities import SQLDatabase
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
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

def get_qdrant_retriever(k):
    qdrant_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return qdrant_retriever
