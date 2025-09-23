# agents/writer_agent.py
# next implement the verification agent

from typing import List
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama
from config import Config

class WriterAgent:
    """
    A specialized agent that acts as the "author".
    Its responsibility is to synthesize a final, coherent answer from the
    reranked and contextually relevant documents provided to it.
    """

    def __init__(self, config: Config, llm: LLM = None):
        """
        Initializes the WriterAgent.

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
                host=config.OLLAMA_HOST
            )
        print("Initialized WriterAgent.")

    async def awrite_answer(self, query: str, reranked_results: List[NodeWithScore],
                            feedback: str = None, previous_answer: str = None) -> str:
        """
        Generates a final answer based on the provided query and reranked context. 
        When feedback is passed from VerifierAgent, rewrite final answer based on suggestions of the feedback in order to correct the previous attempt 'previous_answer'.

        Args:
            query (str): The original user query.
            reranked_results (List[NodeWithScore]): The list of top documents from the RerankerAgent.
            feedback (str, optional): Feedback from the VerifierAgent on a previous answer. 

        Returns:
            str: The synthesized final answer.
        """
        
        if feedback:
            print("\n--- Rewriting final answer based on feedback ---")
        else:
            print(f"\n--- Writing final answer for query: '{query}' ---")

        # Reverse the order to place the most relevant context last, near the query.
        # This is based on the "lost in the middle" phenomenon. (?)
        reversed_results = reranked_results[::-1]

        # Format the context string from the reranked documents.
        context_str = "\n\n".join(
            [f"Source {i+1}:\n{r.node.get_content()}" for i, r in enumerate(reversed_results)]
        )

        feedback_section = ""
        if feedback and previous_answer:
            feedback_section = f"""
        You have previously generated an answer that was found to be unfaithful.
        
        Here was your previous attempt:
        <previous_answer>
        {previous_answer}
        </previous_answer>

        Here is the feedback on what was wrong:
        <feedback>
        {feedback}
        </feedback>

        Please use this feedback to edit and generate a new, corrected, and fully
        faithful answer. Ensure your new answer strictly adheres to the provided
        sources and corrects the identified errors.
        """
        

        prompt = f"""
        You are an expert assistant who synthesizes information to answer questions.
        Your task is to provide a comprehensive and coherent answer to the user's query
        based exclusively on the provided sources.

        First, think step-by-step to identify the key concepts, findings, and data in the sources
        that are relevant to the user's query. Your job is to connect these ideas, even if the
        sources don't use the exact same words as the query. Enclose your entire thought
        process within <think> and </think> tags.

        {feedback_section}

        After your thought process, synthesize your findings into a final, well-structured answer.
        Do not use any external knowledge. If the context is truly insufficient, state that, but
        first make every effort to synthesize the available information.

        Here are the relevant sources:
        <sources>
        {context_str}
        </sources>

        User Query:
        <query>
        {query}
        </query>

        Your final answer should begin after the </think> tag.
        """
        
        response = await self.llm.acomplete(prompt)
        final_answer = response.text

        if "</think>" in response.text:
            final_answer = response.text.split("</think>")[-1].strip()
        
        print("--- Generated Final Answer ---")
        print(final_answer)
        
        return final_answer