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
                base_url=f"http://{config.OLLAMA_HOST}:11434"
            )
        print("Initialized WriterAgent.")

    async def awrite_answer(self, query: str, reranked_results: List[NodeWithScore],
                            feedback: str = None, previous_answer: str = None,
                            guidance: str | None = None,
                            unsupported_sentences: list | None = None,
                            missing_citations: list | None = None,
                            required_facts: list | None = None) -> str:
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

        # Format the context string from the reranked documents such that it includes the sources.
        context_parts = []
        allowed_files = []
        seen_files = set()
        for i, r in enumerate(reversed_results):
            metadata = r.node.metadata or {}
            file_name = metadata.get("source_file") or metadata.get("file_name")
            page_label = metadata.get("source_page_label") or metadata.get("page_label")

            label = f"File: {file_name}" if file_name else "File: Unknown"
            if page_label:
                label += f" (p.{page_label})"

            context_parts.append(f"{label}\n{r.node.get_content()}")
            # Allow any file name (pdf, txt, etc.) so writer citations stay within retrieved context
            if file_name and file_name not in seen_files:
                allowed_files.append(file_name)
                seen_files.add(file_name)

        context_str = "\n\n".join(context_parts)

        feedback_section = ""
        if feedback and previous_answer:
            unsupported_block = "\n".join(f"- {s}" for s in (unsupported_sentences or []))
            missing_block = "\n".join(f"- {s}" for s in (missing_citations or []))
            facts_block = "\n".join(f"- {s}" for s in (required_facts or []))

            feedback_section = f"""
        You have previously generated an answer that needs corrections.
        
        Previous answer:
        <previous_answer>
        {previous_answer}
        </previous_answer>

        Feedback summary:
        <feedback>
        {feedback}
        </feedback>

        Unsupported sentences (fix only these; keep all other sentences unchanged):
        <unsupported>
        {unsupported_block}
        </unsupported>

        Missing citations (add citations to these sentences, without changing their meaning):
        <missing_citations>
        {missing_block}
        </missing_citations>

        Required facts (ensure these exact facts appear with citations):
        <required_facts>
        {facts_block}
        </required_facts>

        Editing policy:
        - If the sentence is supported and already cited, do not change it.
        - If the sentence is unsupported, rewrite only that sentence to be fully grounded by the sources.
        - If only citations are missing, add the correct citation(s) without altering the wording.
        - Preserve the structure and meaning of the previous answer as much as possible.
        - Every sentence in the final answer must end with one or more bracketed citations like [file.pdf, p.X].
        - Use only the information from the Sources block.
        """
        

        # prompt = f"""
        # You are an expert assistant who synthesizes information to answer questions.
        # Your task is to provide a comprehensive and coherent answer to the user's query
        # based exclusively on the provided sources.

        # First, think step-by-step to identify the key concepts, findings, and data in the sources
        # that are relevant to the user's query. Your job is to connect these ideas, even if the
        # sources don't use the exact same words as the query. Enclose your entire thought
        # process within <think> and </think> tags.

        # {feedback_section}

        # After your thought process, generate the final answer by following these MANDATORY rules:
        # 1.  **Strictly Adhere to Sources:** Your answer must be based ONLY on the information present in the <sources> block. Do not use any external knowledge.
        # 2.  **Extract, Don't Interpret:** Your primary task is to extract relevant facts. Directly quote from the sources whenever possible. Do not rephrase, summarize, or synthesize information.
        # 3.  **Cite Every Claim:** Every single statement or piece of information in your answer MUST be followed by a citation in the format `[Source X]`, where X is the number of the source document.
        # 4.  **Handle Insufficient Information:** If the sources do not contain the information needed to answer the query, you MUST state only: "The provided sources do not contain enough information to answer this question."

        # Here are the relevant sources:
        # <sources>
        # {context_str}
        # </sources>

        # User Query:
        # <query>
        # {query}
        # </query>

        # Your final answer should begin after the </think> tag.
        # """

        guidance_section = ""
        if guidance:
            guidance_section = f"""
        Guidance (do not cite):
        <guidance>
        {guidance}
        </guidance>

        Use the guidance only to structure the answer and identify likely facts/terms. Do not quote or cite the guidance; only cite the Sources block.
        """

        allowed_section = ""
        if allowed_files:
            allowed_list = "\n".join(f"- {f}" for f in allowed_files)
            allowed_section = f"""
        Allowed Files to Cite (enforced):
        <allowed_files>
        {allowed_list}
        </allowed_files>

        You must cite ONLY these exact file names. Do not invent new files or use aliases like [Source N]. If a page number is unknown for a file (or the file has no pages, e.g., .txt), cite as [file_name]. When a page is known, use [file_name, p.X].
        """

        prompt = f"""
        You are a meticulous research assistant who writes strictly grounded answers.

        Follow this exact two-step process:
        
        Step 1 — Fact Extraction (notes only)
        - Read the User Query and the Sources.
        - Extract every relevant fact as bullet points.
        - Each bullet MUST end with its citation in the format `[file_name.pdf, p.X]`.
        - Prefer quoting short spans verbatim; avoid paraphrasing when unnecessary.
        
        Step 2 — Grounded Synthesis (final answer only)
        - Write a clear, concise answer composed ONLY of the facts from Step 1.
        - Every sentence MUST end with one or more bracketed citations like `[file_name.pdf, p.X]`. If multiple files or pages support a sentence, include all.
        - Use no external knowledge. No speculation. No uncited claims.
        - If the sources do not contain enough information, output exactly: "The provided sources do not contain enough information to answer this question."

        {feedback_section}
        {guidance_section}
        {allowed_section}
        Perform Step 1 inside a <think> block. Do not include Step 1 in the final answer output.
        
        <think>
        Step 1: Fact Extraction
        - [Extract fact 1 with a short quote if helpful] [source_file.pdf, p.X]
        - [Extract fact 2] [another_file.pdf, p.Y]
        - ...
        
        Step 2: Grounded Synthesis (draft to be cleaned for the final answer)
        [Draft the final answer text here, ensuring every sentence ends with bracketed citations like [file.pdf, p.X].]
        </think>

        Sources:
        <sources>
        {context_str}
        </sources>

        User Query:
        <query>
        {query}
        </query>

        Your final answer should begin after the </think> tag. It must exclude Step 1 and must enforce per-sentence citations.
        """
        
        response = await self.llm.acomplete(prompt)
        final_answer = response.text

        if "</think>" in response.text:
            final_answer = response.text.split("</think>")[-1].strip()
        
        print("--- Generated Final Answer ---")
        print(final_answer)
        
        return final_answer
