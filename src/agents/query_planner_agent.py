# agents/query_planner_agent.py

import json
from typing import List
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
                model=config.UTILITY_MODEL,
                request_timeout=config.LLM_REQUEST_TIMEOUT,
                base_url=f"http://{config.OLLAMA_HOST}:11434"
            )
        print("Initialized QueryPlannerAgent.")

    async def aplan_query(self, query: str, chat_history: List[dict] = None)  -> dict:
        """
        Analyzes the user's query and conversation history to generate a plan.
        It rewrites the query for clarity and conditionally creates a hypothetical
        document only when retrieval is necessary.

        """
    
        print(f"\n--- Planning for query: '{query}' ---")

        chat_history = chat_history or []
        history_formatted = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        
        prompt = f"""
        You are a query analysis and hypothetical document generation agent for a RAG system.
        Your goal is to understand the user's intent in the context of a conversation and
        prepare the query for retrieval.

        Here is the conversation history:
        <history>
        {history_formatted}
        </history>

        Here is the user's latest query:
        <query>
        {query}
        </query>

        **Your Tasks:**

        1.  **Rewrite Query:** First, analyze the history and the latest query. If the query is a follow-up,
            rewrite it to be a standalone, self-contained question. For example, if the user asks "What about its safety features?"
            after a conversation about Llama 2, rewrite it to "What are the safety features of the Llama 2 model?".
            If the query is already standalone, use it as is.

        2.  **Classify Retrieval:** Determine if this rewritten query requires searching an external knowledge base.
            - Retrieval is **not needed** for simple greetings, conversational filler, or basic general knowledge.
            - Retrieval is **required** for queries asking for specific, detailed information.

        3.  **Generate HyDE Document:** Create a concise, one-paragraph hypothetical document that provides a
            plausible, detailed answer to the **rewritten query**. This document will be used to find similar,
            real documents in the knowledge base.

        **Output Format:**
        Provide your response as a single, valid JSON object with three keys:
        - "requires_retrieval": (boolean) Your classification decision.
        - "hyde_document": (string) The hypothetical document.
        - "query": (string) The rewritten, standalone query.

        Do not include any preamble or text outside the JSON object.

        Example format example:
        Query: "What were the key findings of the Llama 2 paper?"
        {{
            "requires_retrieval": true,
            "hyde_document": "The Llama 2 paper introduced a collection of pretrained and fine-tuned large language models (LLMs) ranging from 7 billion to 70 billion parameters. Key findings highlighted its improved performance over the original Llama models, particularly in reasoning, coding, and knowledge-based tasks. The paper also detailed a novel fine-tuning methodology focused on safety and helpfulness, demonstrating that the models could achieve state-of-the-art results while minimizing harmful or biased outputs through techniques like supervised fine-tuning and reinforcement learning with human feedback (RLHF).",
            "query": "What were the key findings of the Llama 2 paper?"
        }}
        """

        print("--- [DEBUG] Prompt constructed. About to call self.llm.acomplete ---")
        response = await self.llm.acomplete(prompt)
        print("--- [DEBUG] LLM call complete. Received response. ---")

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