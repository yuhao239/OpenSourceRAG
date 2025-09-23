# agents/ingestion_agent.py
# This file defines the Ingestion Agent responsible for processing, chunking, and embedding the documents

from config import Config 
import chromadb
import pickle

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

class IngestionAgent():
    """
    A specialized agent responsible for the ingestion part of the RAG pipeline.
    It processes documents, chunks them, embeds them, and stores them in a vector DB.
    """

    def __init__(self, config: Config):
        """
        Initializes the IngestionAgent with the given configuration from the configuration file.
        """
        self.config = config
        self.setup_components()
        print("Initialized IngestionAgent.")


    def setup_components(self):
        """
        Sets up the necessary components based on the configuration. 
        """
        
        # Define the embedding model 
        self.embed_model = HuggingFaceEmbedding(
            model_name = self.config.EMBEDDING_MODEL_NAME
        )

        # Vector Database (chromadb for now)
        db = chromadb.PersistentClient(self.config.CHROMA_PERSIST_DIR)
        chroma_collection = db.get_or_create_collection(self.config.CHROMA_COLLECTION_NAME)
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Ingestion pipeline with sentence splitter and sliding window strategy 
        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=self.config.CHUNK_SIZE,
                    chunk_overlap=self.config.CHUNK_OVERLAP
                ),
                self.embed_model
            ],
            vector_store=self.vector_store
        )

        print("Set up the embedding model, vector store, and ingestion pipeline.")

    async def process_documents(self, documents:list):
        """
        Processes a list of documents through the ingestion pipeline.

        Args: 
            documents (list): A list of llamaindex document objects.
        """
        nodes = await self.pipeline.arun(documents=documents, show_progress=True)
        print("Processed the documents under the data folder.")

        print(f"Saving {len(nodes)} nodes to disk...")
        with open(self.config.NODES_PATH, 'wb') as f:
            pickle.dump(nodes, f)
        print(f"Nodes saved to {self.config.NODES_PATH}")


