# agents/reranker_agent.py
# Reranking phase of the retrieval 

from typing import List 
from config import Config 
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.postprocessor import SentenceTransformerRerank


class RerankerAgent:
    """
    A specialized agent that acts as the "expert analyst".
    Its responsibility is to re-order a list of retrieved documents
    based on their relevance to the original query using a more powerful
    cross-encoder model.
    """

    def __init__(self, config: Config):
        """
        Initializes the RerankerAgent.
        """
        self.config = config

        # Initialize the cross-encoder model for reranking
        # Use a lighter model for now, to be optimized later
        self.reranker = SentenceTransformerRerank(
            model = "cross-encoder/ms-marco-MiniLM-L-2-v2",
            top_n = 3 # Subject to experimentation
        )

    print("Initialized RerankerAgent.")

    async def arerank(self, query: str, documents: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Reranks the given documents against the query.  
        Args:
            query (str): The original user query.
            documents (List[NodeWithScore]): The list of documents from the SearcherAgent.  
        Returns:
            List[NodeWithScore]: The top_n documents, re-ordered and re-scored.
        """

        reranked_results = self.reranker.postprocess_nodes(
            nodes=documents,
            query_str=query
        )

        print(f"Reranking complete. Returning the top {len(reranked_results)} documents.")
        return reranked_results
