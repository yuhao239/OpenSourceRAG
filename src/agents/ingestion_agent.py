# agents/ingestion_agent.py
# This file defines the Ingestion Agent responsible for processing, chunking, and embedding the documents

import os
import pickle

import chromadb
from config import Config
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore


class IngestionAgent:
    """A specialized agent that ingests documents into the vector store."""

    def __init__(self, config: Config):
        self.config = config
        self.setup_components()
        print("Initialized IngestionAgent.")

    @staticmethod
    def clean_text(text: str) -> str:
        import re
        # Remove punctuation, then clean up whitespace
        text_no_punct = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return " ".join(text_no_punct.strip().split())

    def setup_components(self):
        """Sets up embedding model, vector store, and ingestion pipeline."""
        self.embed_model = HuggingFaceEmbedding(
            model_name=self.config.EMBEDDING_MODEL_NAME
        )

        db = chromadb.PersistentClient(self.config.CHROMA_PERSIST_DIR)
        chroma_collection = db.get_or_create_collection(self.config.CHROMA_COLLECTION_NAME)
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=self.config.CHUNK_SIZE,
                    chunk_overlap=self.config.CHUNK_OVERLAP,
                ),
                self.embed_model,
            ],
            vector_store=self.vector_store,
        )

        print("Set up the embedding model, vector store, and ingestion pipeline.")

    @staticmethod
    def find_longest_word(text: str) -> str: 
        """Finds the longest word in a text string to use as a highlight keyword."""
        words = text.split()
        if not words:
            return ""
        return max(words, key=len)
    
    async def process_documents(self, documents: list):
        """Processes documents, embeds them, and saves nodes to disk."""
        nodes = await self.pipeline.arun(documents=documents, show_progress=True)
        print("Processed the documents under the data folder.")

        for node in nodes:
            metadata = node.metadata or {}
            source_path = metadata.get("source_path") or metadata.get("file_path")
            if source_path:
                metadata["source_path"] = source_path
                metadata.setdefault("source_file", os.path.basename(source_path))
            elif metadata.get("file_name"):
                metadata.setdefault("source_file", metadata["file_name"])

            if metadata.get("page_label") and "source_page_label" not in metadata:
                metadata["source_page_label"] = metadata["page_label"]
            if metadata.get("page_number") is not None and "source_page_number" not in metadata:
                metadata["source_page_number"] = metadata["page_number"]

            excerpt_full = self.clean_text(node.get_content())
            if excerpt_full:
                metadata["source_excerpt"] = excerpt_full[:200]
                metadata["highlight_text"] = excerpt_full[:400]
                metadata["highlight_keyword"] = self.find_longest_word(excerpt_full)

            metadata.setdefault("source_node_id", getattr(node, "id_", None))
            node.metadata = metadata

        print(f"Saving {len(nodes)} nodes to disk...")
        with open(self.config.NODES_PATH, "wb") as handle:
            pickle.dump(nodes, handle)
        print(f"Nodes saved to {self.config.NODES_PATH}")
        
        # Return number of processed nodes for reporting
        return len(nodes)
