# agents/searcher_agent.py
# Retriever implemented with hybrid search + hyDE
# Still lacking hybrid search function (BM25)

import chromadb
from typing import List
from config import Config
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import NodeWithScore

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
                model_name=self.config.EMBEDDING_MODEL_NAME,
                device="cuda"
            )
        
        # Connect to the existing vector store
        db = chromadb.PersistentClient(path=self.config.CHROMA_PERSIST_DIR)
        chroma_collection = db.get_collection(self.config.CHROMA_COLLECTION_NAME)
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

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
        query_embedding = self.embed_model.get_text_embedding(hyde_document)
        

        # Execute the search
        retrieval_results = self.vector_store.query(
            VectorStoreQuery(
            query_embedding=query_embedding, 
            similarity_top_k=top_k
            )
        )
        # print(type(retrieval_results.__getattribute__("nodes")[0]))
        # VectorStoreQuery returns a VectorStoreQueryResult which has attribute node of type List[TextNode] and similarity scores of type float
        # Which is why we have to repackage them into a List[NodeWithScore] object that the RerankerAgent is expecting
        nodes_with_scores = []
        for node, score in zip(retrieval_results.nodes, retrieval_results.similarities):
            nodes_with_scores.append(
                NodeWithScore(
                    node=node, 
                    score=score
                )
            )
        
        # The result from ChromaVectorStore is already in the desired format
        # It includes nodes and their scores.
        search_results = nodes_with_scores

        print(f"Found {len(search_results)} documents.")
        return search_results