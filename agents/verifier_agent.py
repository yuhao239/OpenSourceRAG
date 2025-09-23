# agents/verifier_agent.py

import json
from typing import List
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama
from config import Config
import re

class VerifierAgent:
    """
    A specialized agent that acts as the "fact-checker".
    Its responsibility is to verify if the generated answer is faithful
    to the provided source context.
    """

    def __init__(self, config: Config, llm: LLM = None):
        self.config = config
        if llm:
            self.llm = llm
        else:
            self.llm = Ollama(
                model=config.LLM_MODEL,
                request_timeout=config.LLM_REQUEST_TIMEOUT,
                host=config.OLLAMA_HOST
            )
        print("Initialized VerifierAgent.")

    async def averify_answer(
        self, query: str, generated_answer: str, source_context: List[NodeWithScore]
    ) -> dict:
        """
        Verifies the generated answer against the source context for faithfulness.

        Args:
            query (str): The original user query.
            generated_answer (str): The answer produced by the WriterAgent.
            source_context (List[NodeWithScore]): The source documents.

        Returns:
            dict: A dictionary containing the verification result:
                  - 'is_faithful': A boolean.
                  - 'feedback': A string with the reasoning.
        """
        print("\n--- Verifying final answer for faithfulness ---")

        context_str = "\n\n".join(
            [f"Source {i+1}:\n{r.node.get_content()}" for i, r in enumerate(source_context)]
        )

        prompt = f"""
        You are a meticulous fact-checking agent. Your task is to verify if the claims
        in a generated answer are fully supported by a given set of source documents.

        Here is the original query, the sources, and the generated answer.

        <sources>
        {context_str}
        </sources>

        <query>
        {query}
        </query>

        <generated_answer>
        {generated_answer}
        </generated_answer>

        Analyze the generated answer claim by claim. Compare each claim against the
        information present in the sources.

        Respond with a single, valid JSON object with two keys:
        1. "is_faithful": A boolean value. Set to `true` ONLY if every single claim in the
           generated answer is directly and explicitly supported by the sources. Otherwise,
           set to `false`.
        2. "feedback": A string providing a brief explanation for your decision. If the
           answer is not faithful, specify which claim is unsupported or misaligned with
           the sources.

        Do not include any preamble or text outside of the JSON object.
        """

        response = await self.llm.acomplete(prompt)
        
        try:
            # Use regex to find the JSON object within the response text
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                json_string = json_match.group()
                result = json.loads(json_string)
            else:
                raise json.JSONDecodeError("No JSON object found in response.", response.text, 0)
            
            print(f"Verification result: Is Faithful? {result.get('is_faithful')}")
            print(f"Feedback: {result.get('feedback')}")
            return result
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing VerifierAgent response: {e}")
            print(f"Raw response: {response.text}")
            # Fallback in case of error
            return {
                "is_faithful": False,
                "feedback": "Could not parse the verification result from the LLM.",
            }