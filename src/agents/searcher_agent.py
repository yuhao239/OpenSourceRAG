# agents/searcher_agent.py
# Retriever implemented with hybrid search + HyDE

import asyncio
import pickle
from typing import List

import chromadb

from config import Config
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore


class SearcherAgent:
    """
    A specialized agent that acts as the "librarian".
    Its responsibility is to retrieve relevant documents from the vector database
    based on a query and a hypothetical document.
    """

    def __init__(self, config: Config, embed_model=None):
        """Initializes the SearcherAgent."""
        self.config = config

        self.embed_model = embed_model or HuggingFaceEmbedding(
            model_name=self.config.EMBEDDING_MODEL_NAME
        )

        db = chromadb.PersistentClient(path=self.config.CHROMA_PERSIST_DIR)
        chroma_collection = db.get_or_create_collection(self.config.CHROMA_COLLECTION_NAME)
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        self.bm25_retriever = None
        try:
            with open(self.config.NODES_PATH, 'rb') as handle:
                nodes = pickle.load(handle)
                if nodes:
                    self.bm25_retriever = BM25Retriever.from_defaults(
                        nodes=nodes,
                        similarity_top_k=5
                    )
                    print("Loaded nodes and initialized BM25 retriever.")
        except FileNotFoundError:
            print("Nodes file not found. Run ingestion before enabling BM25 hybrid search.")
        except Exception as exc:
            print(f"Failed to initialize BM25 retriever: {exc}")

        print("Initialized SearcherAgent.")

    async def asearch(self, query: str, hyde_document: str, top_k: int = 5) -> List[NodeWithScore]:
        """Performs hybrid search against the knowledge base."""
        print(f"\n--- Searching for documents related to: '{query}' ---")

        query_embedding = await self.embed_model.aget_text_embedding(hyde_document)

        vector_results = self.vector_store.query(
            VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=top_k
            )
        )

        nodes_with_scores: List[NodeWithScore] = []
        for node, score in zip(vector_results.nodes or [], vector_results.similarities or []):
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        bm25_results: List[NodeWithScore] = []
        if self.bm25_retriever is not None:
            bm25_results = await asyncio.to_thread(self.bm25_retriever.retrieve, query)
        else:
            print("BM25 retriever is not initialized; returning vector results only.")

        search_results = {}
        for result in nodes_with_scores + bm25_results:
            search_results[result.id_] = result

        print(f"Search complete. Found {len(search_results)} unique documents.")
        return list(search_results.values())




