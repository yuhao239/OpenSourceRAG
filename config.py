# config.py
"""
This file contains the configuration settings for our RAG pipeline.
It's updated to read database settings from environment variables for Docker compatibility.
"""
import os

class Config():
    """Configuration settings for RAG pipeline."""

    DATA_DIR = "./data"
    CHROMA_PERSIST_DIR = "./db"
    CHROMA_COLLECTION_NAME = "best_practices_rag"
    NODES_PATH = "./db/nodes.pkl"

    # --- Chunking Settings ---
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64

    # --- Embedding model ---
    EMBEDDING_MODEL_NAME = "BAAI/llm-embedder"

    # --- LLM settings ---
    LLM_MODEL = "qwen3:8b"
    UTILITY_MODEL = "qwen3:1.7b"
    LLM_REQUEST_TIMEOUT = 120.0
    LLM_CONTEXT_WINDOW = 8000
    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "localhost")

    # --- Agent settings ---
    USE_VERIFIER = True
    MAX_REWRITES = 2

    # --- PostgreSQL Database Connection ---
    # Read from environment variables if they exist (for Docker),
    # otherwise use default values for local development.
    db_host = os.environ.get("DB_HOST", "localhost")
    db_user = os.environ.get("DB_USER", "rag_user")
    db_password = os.environ.get("DB_PASSWORD", "mysecretpassword")
    db_name = os.environ.get("DB_NAME", "rag_db")
    db_port = os.environ.get("DB_PORT", "5432")

    db_connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

