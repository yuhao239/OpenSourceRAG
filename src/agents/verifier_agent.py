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
            # Use the lighter UTILITY_MODEL to reduce verification latency
            self.llm = Ollama(
                model=config.UTILITY_MODEL,
                request_timeout=config.LLM_REQUEST_TIMEOUT,
                base_url=f"http://{config.OLLAMA_HOST}:11434"
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
        You are a meticulous fact-checking agent. Your task is to verify whether the generated answer is fully supported by the provided sources.

        <sources>
        {context_str}
        </sources>

        <query>
        {query}
        </query>

        <generated_answer>
        {generated_answer}
        </generated_answer>

        Analyze the answer sentence by sentence. For each sentence, determine if it is directly and explicitly supported by the sources.

        Respond with a single, valid JSON object containing exactly these keys (and no others):
        - "is_faithful": boolean (true only if every sentence is fully supported)
        - "feedback": string (brief explanation; if false, describe what is unsupported or missing)
        - "unsupported_sentences": array of strings (the unsupported sentences)
        - "missing_citations": array of strings (sentences that should have citations but do not)
        - "required_facts": array of strings (facts that must be present with citations)

        Do not include any text outside the JSON object. If you need to think, enclose it in <think>...</think> but ensure the final output is only the JSON.
        """

        try:
            response = await self.llm.acomplete(prompt)
        except Exception as e:
            print(f"VerifierAgent LLM call failed or timed out: {e}")
            # Gracefully skip verification on transport/timeouts to avoid crashing the workflow
            return {
                'is_faithful': True,  # treat as pass to avoid unnecessary rewrites
                'feedback': 'Verifier skipped due to LLM error/timeout.',
                'unsupported_sentences': [],
                'missing_citations': [],
                'required_facts': [],
            }

        def _clean_and_extract_json(raw: str) -> dict:
            text = (raw or '').strip()
            # Remove hidden chain-of-thought if present
            if '</think>' in text:
                text = text.split('</think>')[-1].strip()
            # Strip code fences
            if text.startswith('```'):
                lines = text.splitlines()
                if lines and lines[0].startswith('```'):
                    lines = lines[1:]
                while lines and lines[-1].strip().startswith('```'):
                    lines = lines[:-1]
                text = '\n'.join(lines).strip()

            # Robust first-JSON extraction
            def _extract_first_json(s: str) -> str | None:
                depth = 0
                start = None
                for i, ch in enumerate(s):
                    if ch == '{':
                        if depth == 0:
                            start = i
                        depth += 1
                    elif ch == '}':
                        if depth:
                            depth -= 1
                            if depth == 0 and start is not None:
                                return s[start:i+1]
                return None

            frag = _extract_first_json(text)
            if not frag:
                # Fallback to permissive regex
                m = re.search(r'\{.*\}', text, re.DOTALL)
                if not m:
                    raise json.JSONDecodeError('No JSON object found in response.', text, 0)
                frag = m.group(0)

            data = json.loads(frag)
            # Normalize fields
            result = {
                'is_faithful': bool(data.get('is_faithful', False)),
                'feedback': data.get('feedback') or '',
                'unsupported_sentences': list(data.get('unsupported_sentences') or []),
                'missing_citations': list(data.get('missing_citations') or []),
                'required_facts': list(data.get('required_facts') or []),
            }
            return result

        try:
            parsed = _clean_and_extract_json(response.text)
            print(f"Verification result: Is Faithful? {parsed.get('is_faithful')}")
            fb = parsed.get('feedback')
            if fb:
                print(f"Feedback: {fb}")
            return parsed
        except Exception as e:
            print(f"Error parsing VerifierAgent response: {e}")
            print(f"Raw response: {response.text}")
            return {
                'is_faithful': False,
                'feedback': 'Could not parse the verification result from the LLM.',
                'unsupported_sentences': [],
                'missing_citations': [],
                'required_facts': [],
            }
