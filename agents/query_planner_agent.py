# agents/query_planner_agent.py

import json
from llama_index.core.llms import LLM
from llama_index.llms.ollama import Ollama
from config import Config

class QueryPlannerAgent:
    """
    A specialized agent that acts as the "gatekeeper" and "strategist".
    It performs two main tasks:
    1.  Query Classification: Determines if retrieval from the knowledge base is necessary.
    2.  HyDE Generation: Creates a hypothetical document to aid in retrieval.
    """

    def __init__(self, config: Config, llm: LLM = None):
        """
        Initializes the QueryPlannerAgent.
        Args:
            config (Config): The application configuration object.
            llm (LLM, optional): An instance of a LlamaIndex LLM. Defaults to None.
        """
        self.config = config
        if llm:
            self.llm = llm
        else:
            self.llm = Ollama(
                model=config.LLM_MODEL,
                request_timeout=config.LLM_REQUEST_TIMEOUT,
            )
        print("Initialized QueryPlannerAgent.")

    async def aplan_query(self, query: str) -> dict:
        """
        Analyzes the user's query and generates a plan for the subsequent agents.

        Args:
            query (str): The user's input query.

        Returns:
            dict: A dictionary containing the plan:
                  - 'requires_retrieval': A boolean indicating if DB search is needed.
                  - 'hyde_document': A generated hypothetical document.
                  - 'query': The original user query.
        """
        print(f"\n--- Planning for query: '{query}' ---")
        
        prompt = f"""
        You are a query classification and hypothetical document generation agent.
        Your goal is to prepare for a Retrieval-Augmented Generation (RAG) system.

        Analyze the following user query:
        <query>
        {query}
        </query>

        First, determine if this query requires searching an external knowledge base to answer.
        - If the query is a simple greeting, conversational, or a question that can be answered with general knowledge (e.g., "What is 2+2?"), retrieval is not needed.
        - If the query asks for specific information, data, or details that are likely contained in a database, retrieval is required.

        Second, generate a concise, one-paragraph hypothetical document that provides a plausible, detailed answer to the query. This document will be used to find similar, real documents in the knowledge base.

        Provide your response as a single, valid JSON object with three keys: "requires_retrieval" (boolean), "hyde_document" (string), and "query" (string, the original query).
        Do not include any preamble, explanations, or markdown formatting outside of the JSON object.

        Example:
        Query: "What were the key findings of the Llama 2 paper?"
        {{
            "requires_retrieval": true,
            "hyde_document": "The Llama 2 paper introduced a collection of pretrained and fine-tuned large language models (LLMs) ranging from 7 billion to 70 billion parameters. Key findings highlighted its improved performance over the original Llama models, particularly in reasoning, coding, and knowledge-based tasks. The paper also detailed a novel fine-tuning methodology focused on safety and helpfulness, demonstrating that the models could achieve state-of-the-art results while minimizing harmful or biased outputs through techniques like supervised fine-tuning and reinforcement learning with human feedback (RLHF).",
            "query": "What were the key findings of the Llama 2 paper?"
        }}
        """

        response = await self.llm.acomplete(prompt)
        
        try:
            # LLM response contains excess <think> section that is messing with extracting the json 
            # So format the response to contain only the json file 
            json_start = response.text.find('{')
            json_end = response.text.find('}') + 1 

            if json_start != -1 and json_end != 0:
                json_string = response.text[json_start:json_end]
                plan = json.loads(json_string)
            else:
                raise json.JSONDecodeError("No JSON object found in response.", response.text, 0)

            print(f"Query requires retrieval: {plan.get('requires_retrieval')}")
            print(f"Generated HyDE document (preview): '{plan.get('hyde_document', '')[:100]}...'")
            return plan
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw response: {response.text}")
            # Fallback in case of parsing error
            return {
                "requires_retrieval": True,
                "hyde_document": "Could not generate a hypothetical document.",
                "query": query,
            }