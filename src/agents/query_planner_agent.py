# agents/query_planner_agent.py

import json
from typing import List, Optional

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

    async def aplan_query(self, query: str, chat_history: Optional[List[dict]] = None) -> dict:
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
            - Retrieval is **not needed** for simple greetings, conversational filler, or anything that you would consider within the scope of general knowledge.
            - Retrieval is **required** for queries asking for specific, detailed information.
            - If the query references a specific paper, model, dataset, release/version, or asks for sources/citations (e.g., "where did you find that"), set `requires_retrieval` to true.
            - When uncertain, prefer `requires_retrieval: true`.

        3.  **Generate HyDE Document:** Create a concise, single-paragraph hypothetical document (120–200 words)
            in the same style as the corpus (formal/academic). Include exact entities, acronyms, synonyms and likely
            section cues (e.g., Abstract, Model Architecture) that would appear in a real document. This document
            will be used to find similar, real documents in the knowledge base.

        **Output Format:**
        Provide your response as a single, valid JSON object with exactly these keys:
        - "requires_retrieval": (boolean) Your classification decision.
        - "hyde_document": (string) The hypothetical document.
        - "query": (string) The rewritten, standalone query.

        Constraints:
        - The "query" field MUST be a rewrite of the user's latest query above, not an example.
        - The "query" must contain at least one non-stopword from the user's latest query; otherwise set it equal to the user's latest query.
        - Do not include any preamble or examples. Output ONLY the JSON object.

        Do not include examples in your output.
        """

        print("--- [DEBUG] Prompt constructed. About to call self.llm.acomplete ---")
        try:
            response = await self.llm.acomplete(prompt)
        except Exception as exc:
            error_text = str(exc)
            print(f"QueryPlannerAgent encountered an error from Ollama: {error_text}")
            lowercase_error = error_text.lower()
            model_keywords = (
                'not found',
                'no such model',
                'model not found',
                'missing model',
                ' 404',
            )
            if any(keyword in lowercase_error for keyword in model_keywords):
                raise RuntimeError(
                    f"Ollama could not load the utility model '{self.config.UTILITY_MODEL}'. "
                    "Make sure it is pulled inside the Ollama container, e.g. "
                    f"`docker compose exec ollama ollama pull {self.config.UTILITY_MODEL}`."
                ) from exc
            raise
        print("--- [DEBUG] LLM call complete. Received response. ---")

        try:
            raw_text = response.text.strip()
            if "</think>" in raw_text:
                raw_text = raw_text.split("</think>")[-1].strip()

            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                while lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines).strip()

            def _extract_first_json(text: str) -> Optional[str]:
                depth = 0
                start = None
                for idx, ch in enumerate(text):
                    if ch == '{':
                        if depth == 0:
                            start = idx
                        depth += 1
                    elif ch == '}':
                        if depth:
                            depth -= 1
                            if depth == 0 and start is not None:
                                return text[start:idx + 1]
                return None

            json_fragment = _extract_first_json(raw_text)
            if not json_fragment:
                raise json.JSONDecodeError("No JSON object found in response.", raw_text, 0)

            plan = json.loads(json_fragment)

            # --- Retrieval-bias heuristics to correct LLM misclassification ---
            # Favor retrieval for domain/document-grounded questions and citation requests.
            def _tokens(text: str) -> set[str]:
                import re
                if not text:
                    return set()
                toks = [t for t in re.split(r"\W+", text) if t]
                upp = re.findall(r"\b[A-Z]{2,}\b", text)
                nums = re.findall(r"\b\d+(?:\.\d+)?\b", text)
                alnum = re.findall(r"\b[a-zA-Z]*\d+[a-zA-Z0-9]*\b", text)
                base = [t.lower() for t in toks if len(t) >= 3]
                base.extend([t.lower() for t in upp])
                base.extend([t.lower() for t in nums])
                base.extend([t.lower() for t in alnum if len(t) >= 2])
                return set(base)

            def _collect_kb_tokens() -> set[str]:
                import os, re, pickle
                kb = set()
                # Prefer filenames from data dir
                try:
                    from config import Config as _Cfg
                    if os.path.isdir(_Cfg.DATA_DIR):
                        for name in os.listdir(_Cfg.DATA_DIR):
                            base = os.path.splitext(name)[0]
                            for t in re.split(r"\W+", base.lower()):
                                if len(t) >= 3:
                                    kb.add(t)
                except Exception:
                    pass
                # Also try node metadata if available
                try:
                    from config import Config as _Cfg
                    if os.path.exists(_Cfg.NODES_PATH):
                        with open(_Cfg.NODES_PATH, 'rb') as h:
                            nodes = pickle.load(h)
                        for n in nodes or []:
                            md = getattr(n, 'metadata', {}) or {}
                            f = (md.get('source_file') or md.get('file_name') or '')
                            base = os.path.splitext(f)[0]
                            for t in re.split(r"\W+", base.lower()):
                                if len(t) >= 3:
                                    kb.add(t)
                except Exception:
                    pass
                return kb

            q = (plan.get('query') or query or '').strip()
            q_toks = _tokens(q)
            kb_toks = _collect_kb_tokens()
            kb_hit = bool(q_toks & kb_toks)

            # Phrases that imply document grounding/citations
            lower_q = q.lower()
            doc_phrases = (
                'paper', 'arxiv', 'pdf', 'section', 'abstract', 'figure', 'table',
                'where did you find', 'source', 'sources', 'cite', 'citation', 'reference'
            )
            wants_citation = any(p in lower_q for p in doc_phrases)

            # Lightweight greeting / small-talk detector
            smalltalk = any(t in {'hi','hello','hey','thanks','thank','bye'} for t in q_toks)

            # Heuristic rule: if references to papers/sources or KB tokens detected, force retrieval
            # Otherwise, if clearly small-talk, allow non-retrieval; when uncertain, prefer retrieval
            heur_requires_retrieval = wants_citation or kb_hit or (not smalltalk)
            original = plan.get('requires_retrieval')
            if heur_requires_retrieval and not original:
                plan['requires_retrieval'] = True
                plan['heuristic_reason'] = 'kb_or_citation_cue'
            elif original is None:
                plan['requires_retrieval'] = True
                plan['heuristic_reason'] = 'default_true'

            print(f"Query requires retrieval: {plan.get('requires_retrieval')}")
            print(f"Generated HyDE document (preview): '{plan.get('hyde_document', '')[:100]}...'")
            return plan
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"Error parsing LLM response: {exc}")
            print(f"Raw response: {response.text}")
            return {
                "requires_retrieval": True,
                "hyde_document": "Could not generate a hypothetical document.",
                "query": query,
            }


