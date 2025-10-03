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
    CHUNK_OVERLAP = 128

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
    MAX_REWRITES = 1

    # --- Evidence-anchored citation thresholds (BM25) ---
    # Minimum number of shared tokens between sentence and chosen node
    CITATION_BM25_MIN_OVERLAP = int(os.environ.get("CITATION_BM25_MIN_OVERLAP", 2))
    # Required margin ratio over second-best score (if MIN_ABS not met)
    CITATION_BM25_MARGIN_RATIO = float(os.environ.get("CITATION_BM25_MARGIN_RATIO", 1.2))
    # Minimum absolute BM25 score to accept without margin
    CITATION_BM25_MIN_ABS = float(os.environ.get("CITATION_BM25_MIN_ABS", 1.5))

    # --- Source packaging / coverage ---
    # After building sentence-aligned, page-biased sources, append this many
    # generic top sources to improve evaluator coverage.
    CITED_SOURCES_COVERAGE_TAIL = int(os.environ.get("CITED_SOURCES_COVERAGE_TAIL", 2))

    # Add a small number of uncited sentence supports by attaching best-matching
    # snippets even when a sentence lacks brackets, to avoid sparse judged context.
    UNCITED_SENTENCE_SUPPORT = int(os.environ.get("UNCITED_SENTENCE_SUPPORT", 2))

    # --- Retrieval confidence gating ---
    # Evaluate retrieval results quality to decide if we should proceed with RAG.
    RETR_CONF_TOPK = int(os.environ.get("RETR_CONF_TOPK", 5))
    RETR_CONF_MIN_HITS = int(os.environ.get("RETR_CONF_MIN_HITS", 2))
    RETR_CONF_MIN_SCORE = float(os.environ.get("RETR_CONF_MIN_SCORE", 0.30))
    RETR_CONF_MIN_OVERLAP = int(os.environ.get("RETR_CONF_MIN_OVERLAP", 3))
    RETR_CONF_MEAN_OVERLAP = float(os.environ.get("RETR_CONF_MEAN_OVERLAP", 2.0))

    # --- PostgreSQL Database Connection ---
    # Read from environment variables if they exist (for Docker),
    # otherwise use default values for local development.
    db_host = os.environ.get("DB_HOST", "localhost")
    db_user = os.environ.get("DB_USER", "rag_user")
    db_password = os.environ.get("DB_PASSWORD", "mysecretpassword")
    db_name = os.environ.get("DB_NAME", "rag_db")
    db_port = os.environ.get("DB_PORT", "5432")

    db_connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    cors_origins_env = os.environ.get("CORS_ALLOW_ORIGINS")
    if cors_origins_env:
        CORS_ALLOW_ORIGINS = [
            origin.strip()
            for origin in cors_origins_env.split(',')
            if origin.strip()
        ] or ['*']
    else:
        CORS_ALLOW_ORIGINS = [
            'http://localhost:8501',
            'http://127.0.0.1:8501',
        ]


