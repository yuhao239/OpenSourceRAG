# agents/searcher_agent.py
# Retriever implemented with hybrid search + hyDE
# Still lacking hybrid search function (BM25)

import chromadb
import asyncio
import pickle
from typing import List
from config import Config
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever

class SearcherAgent:
    """
    A specialized agent that acts as the "librarian".
    Its responsibility is to retrieve relevant documents from the vector database
    based on a query and a hypothetical document.
    """

    def __init__(self, config: Config, embed_model=None):
        """
        Initializes the SearcherAgent.
        """
        self.config = config
        
        # Initialize the embedding model
        if embed_model:
            self.embed_model = embed_model
        else:
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.config.EMBEDDING_MODEL_NAME
            )
        
        # Connect to the existing vector store
        db = chromadb.PersistentClient(path=self.config.CHROMA_PERSIST_DIR)
        chroma_collection = db.get_collection(self.config.CHROMA_COLLECTION_NAME)
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        try:
            with open(self.config.NODES_PATH, 'rb') as f:
                nodes = pickle.load(f)
                self.bm25_retrievers = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
                print("Loaded nodes and initialized BM25.")

        except FileNotFoundError:
            print("Nodes file not found, ensure they have processed and saved to disk.")
            self.bm25_retrievers = None

        print("Initialized SearcherAgent.")

    async def asearch(self, query: str, hyde_document: str, top_k: int = 5) -> List[NodeWithScore]:
        """
        Performs vector search on the ChromaDB vector store.

        Args:
            query (str): The original user query.
            hyde_document (str): The hypothetical document from the QueryPlannerAgent.
            top_k (int): The number of top results to retrieve.

        Returns:
            List[NodeWithScore]: A list of retrieved nodes with their similarity scores.
        """
        print(f"\n--- Searching for documents related to: '{query}' ---")
        
        # Embed the hypothetical document to create the query embedding
        query_embedding = await self.embed_model.aget_text_embedding(hyde_document)
        
        # Execute the search
        vector_results = self.vector_store.query(
            VectorStoreQuery(
            query_embedding=query_embedding, 
            similarity_top_k=top_k
            )
        )
        # print(type(retrieval_results.__getattribute__("nodes")[0]))
        # VectorStoreQuery returns a VectorStoreQueryResult which has attribute node of type List[TextNode] and similarity scores of type float
        # Which is why we have to repackage them into a List[NodeWithScore] object that the RerankerAgent is expecting
        nodes_with_scores = []
        for node, score in zip(vector_results.nodes, vector_results.similarities):
            nodes_with_scores.append(
                NodeWithScore(
                    node=node, 
                    score=score
                )
            )

        bm25_results = await asyncio.to_thread(self.bm25_retrievers.retrieve, query)

        search_results = {} # Use a dict for ensuring uniqueness of documents

        for node in nodes_with_scores + bm25_results:
            search_results[node.id_] = node

        print(f"Search complete. Found {len(search_results)} documents.")
        return list(search_results.values())
        