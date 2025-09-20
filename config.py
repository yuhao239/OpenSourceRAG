# config.py
# This file contains the configuration settings for our RAG pipeline

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

    # --- LLM settings (settings to be tweaked later on) --- 
    LLM_MODEL = "qwen3:8b"
    LLM_REQUEST_TIMEOUT = 60.0
    LLM_CONTEXT_WINDOW = 8000 

    # --- Agent settings --- 
    USE_VERIFIER = True # Set to False to disable the VerifierAgent
    MAX_REWRITES = 2