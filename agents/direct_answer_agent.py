# agents/direct_answer_agent.py

from llama_index.core.llms import LLM
from llama_index.llms.ollama import Ollama
from config import Config

class DirectAnswerAgent:
    """
    A specialized agent that provides direct, concise answers to queries
    that do not require document retrieval, using a HyDE doc for context.
    """
    def __init__(self, config: Config, llm: LLM = None):
        self.config = config
        if llm:
            self.llm = llm
        else:
            self.llm = Ollama(
                model=config.LLM_MODEL,
                request_timeout=config.LLM_REQUEST_TIMEOUT,
            )
        print("Initialized DirectAnswerAgent.")

    async def adirect_answer(self, query: str, documents: str) -> str:
        """
        Generates a direct answer using the HyDE document as context.
        """
        prompt = f"""
        You are a helpful and friendly conversational assistant.
        Use the following context document to provide a brief and direct answer to the user's query.
        If the user is making a simple greeting, ignore the context and respond naturally.
        If the user is asking a general knowledge question, use the context to provide a concise answer.
        If you need to think, enclose your thought process in <think></think> tags. Don't think unless necessary.

        Context Document:
        "{documents}"

        User Query:
        "{query}"
        """
        
        response = await self.llm.acomplete(prompt)
        raw_text = response.text.strip()
        
        final_answer = raw_text
        if "</think>" in raw_text:
            final_answer = raw_text.split("</think>")[-1].strip()
        
        return final_answer