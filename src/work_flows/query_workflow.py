import os
import time
# workflows/query_workflow.py
# Workflow orchestrating the whole query to answer pipeline

from .base import Workflow
from events import (
    StartQueryEvent,
    QueryPlanningCompleteEvent,
    StopEvent,
    SearchCompleteEvent,
    RerankCompleteEvent,
    WritingCompleteEvent,
    RewriteEvent
)
from agents.query_planner_agent import QueryPlannerAgent
from config import Config
from agents.searcher_agent import SearcherAgent
from agents.reranker_agent import RerankerAgent
from agents.writer_agent import WriterAgent
from agents.verifier_agent import VerifierAgent
from agents.direct_answer_agent import DirectAnswerAgent


class QueryWorkflow(Workflow):
    """Orchestrates the real-time, interactive query process."""

    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.query_planner_agent = QueryPlannerAgent(config)
        self.searcher_agent = SearcherAgent(config)
        self.reranker_agent = RerankerAgent(config)
        self.writer_agent = WriterAgent(config)
        self.verifier_agent = VerifierAgent(config)
        self.direct_answer_agent = DirectAnswerAgent(config)

        self.add_listener(StartQueryEvent, self.start_query_planning)
        self.add_listener(QueryPlanningCompleteEvent, self.start_search)
        self.add_listener(SearchCompleteEvent, self.start_reranking)
        self.add_listener(RerankCompleteEvent, self.handle_writing_request)
        self.add_listener(RewriteEvent, self.handle_writing_request)
        self.add_listener(WritingCompleteEvent, self.start_verification)

    def _set_status(self, phase: str) -> None:
        callback = self.context.get('set_status')
        if callable(callback):
            try:
                callback(phase)
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[Workflow Status] Failed to update status: {exc}")

    def collect_sources(self, nodes):
        sources = []
        seen = set()
        import re
        query_text = (self.context.get('last_query') or '').strip()
        stopwords = {
            'the','a','an','and','or','but','if','then','else','for','to','of','in','on','at','by','with',
            'is','are','was','were','be','been','being','this','that','these','those','it','as','from','about',
            'what','which','who','whom','how','why','when','where','can','could','should','would','do','does','did'
        }
        query_tokens = []
        if query_text:
            query_tokens = [t for t in re.split(r"\W+", query_text) if t and t.lower() not in stopwords]
            query_tokens.sort(key=lambda s: (-len(s), s.lower()))

        def choose_search_term(raw_text: str) -> str | None:
            if not raw_text:
                return None
            lower_text = raw_text.lower()
            for qt in query_tokens:
                if qt.lower() in lower_text and len(qt) >= 4:
                    return qt
            m = re.search(r"[A-Za-z][A-Za-z0-9\.-]{4,}", raw_text)
            if m:
                return m.group(0)
            words = [w for w in re.split(r"\s+", raw_text.strip()) if w]
            if words:
                return " ".join(words[:3])[:80]
            return None
        for idx, node_with_score in enumerate(nodes or [], start=1):
            node = node_with_score.node
            metadata = getattr(node, "metadata", {}) or {}
            source_path = metadata.get("source_path") or metadata.get("file_path")
            source_file = metadata.get("source_file") or (
                os.path.basename(source_path) if source_path else None
            )
            if not source_file:
                continue

            raw_page_number = metadata.get("source_page_number")
            page_label = metadata.get("source_page_label") or metadata.get("page_label")
            # Compute a reliable 1-based page number for PDF viewers
            page_number = None
            if raw_page_number is not None:
                try:
                    pn = int(raw_page_number)
                    page_number = pn + 1 if pn <= 0 else pn
                except Exception:
                    page_number = None
            if page_number is None and page_label is not None:
                try:
                    if str(page_label).isdigit():
                        page_number = int(page_label)
                except Exception:
                    page_number = None
            raw_content = node.get_content()
            excerpt = metadata.get("source_excerpt") or (raw_content.strip()[:200] if raw_content else None)
            highlight_id = metadata.get("source_node_id") or getattr(node, "id_", None)
            highlight_keyword = metadata.get("highlight_keyword") or choose_search_term(raw_content)

            key = (source_file, page_number, highlight_id)
            if key in seen:
                continue
            seen.add(key)

            sources.append({
                "id": highlight_id,
                "file": source_file,
                "path": source_path,
                "page": page_number,
                "page_label": page_label,
                "score": float(node_with_score.score) if node_with_score.score is not None else None,
                "excerpt": excerpt,
                "highlight_text": metadata.get("highlight_text") or excerpt,
                "highlight_keyword": highlight_keyword,
                "rank": idx,
            })
        return sources

    async def start_query_planning(self, event: StartQueryEvent):
        """Entry point for the query workflow."""
        print("\n--- Query Workflow Started ---")
        self._set_status('Planning query')

        self.context['rewrite_cycles'] = 0
        self.context['timings'] = {}
        self.context['source_refs'] = []
        chat_history = self.context.get('chat_history', [])
        self.context['workflow_start_time'] = time.monotonic()
        step_start_time = time.monotonic()

        try:
            plan = await self.query_planner_agent.aplan_query(event.query, chat_history=chat_history)
            self.context['timings']['query_planning'] = time.monotonic() - step_start_time
            self.context['last_query'] = plan.get("query") or event.query
        except Exception as exc:  # pragma: no cover - defensive guardrail
            self.context['timings']['query_planning'] = time.monotonic() - step_start_time
            self._set_status('Failed during planning')
            error_message = str(exc)
            print(f"Query planning failed: {error_message}")
            await self.dispatch(
                StopEvent(
                    result={
                        'error': error_message,
                        'phase': 'query_planning',
                        'timings': self.context.get('timings', {}),
                    }
                )
            )
            return

        await self.dispatch(
            QueryPlanningCompleteEvent(
                query=plan.get("query"),
                hyde_document=plan.get("hyde_document"),
                requires_retrieval=plan.get("requires_retrieval", True),
                result=plan
            )
        )
    async def start_search(self, ev: QueryPlanningCompleteEvent):
        """Triggered by QueryPlanningCompleteEvent. Runs the SearcherAgent."""
        if not ev.requires_retrieval:
            print("\n--- Retrieval not required. Skipping Search. ---")
            self._set_status('Answering directly')

            step_start_time = time.monotonic()
            final_answer = await self.direct_answer_agent.adirect_answer(
                query=ev.query,
                documents=ev.hyde_document
            )
            self.context['timings']['direct_answer'] = time.monotonic() - step_start_time
            self.context['timings']['total_workflow'] = time.monotonic() - self.context['workflow_start_time']
            self.context['source_refs'] = []
            self._set_status('Completed')
            await self.dispatch(StopEvent(result={
                "final_answer": final_answer,
                "verification_feedback": "N/A (Direct Answer)",
                'timings': self.context['timings'],
                'sources': self.context['source_refs']
            }))
            return

        step_start_time = time.monotonic()
        self._set_status('Searching knowledge base')
        search_results = await self.searcher_agent.asearch(
            query=ev.query,
            hyde_document=ev.hyde_document
        )
        self.context['timings']['search'] = time.monotonic() - step_start_time

        await self.dispatch(
            SearchCompleteEvent(
                query=ev.query,
                search_results=search_results
            )
        )

    async def start_reranking(self, ev: SearchCompleteEvent):
        """Triggered by SearchCompleteEvent. Runs the RerankerAgent."""
        step_start_time = time.monotonic()
        self._set_status('Reranking results')
        reranked_results = await self.reranker_agent.arerank(
            query=ev.query,
            documents=ev.search_results
        )
        self.context['timings']['reranking'] = time.monotonic() - step_start_time

        await self.dispatch(
            RerankCompleteEvent(
                query=ev.query,
                reranked_results=reranked_results
            )
        )

    async def handle_writing_request(self, ev: RerankCompleteEvent | RewriteEvent):
        """Handles both initial writing and rewrites."""
        feedback = ev.feedback if isinstance(ev, RewriteEvent) else None
        previous_answer = ev.previous_answer if isinstance(ev, RewriteEvent) else None

        self.context['source_refs'] = self.collect_sources(ev.reranked_results)

        phase_label = 'Rewriting answer' if feedback else 'Composing answer'
        self._set_status(phase_label)

        step_start_time = time.monotonic()
        generated_answer = await self.writer_agent.awrite_answer(
            query=ev.query,
            reranked_results=ev.reranked_results,
            feedback=feedback,
            previous_answer=previous_answer
        )
        self.context['timings']['writing'] = self.context['timings'].get('writing', 0) + (time.monotonic() - step_start_time)

        await self.dispatch(
            WritingCompleteEvent(
                query=ev.query,
                reranked_results=ev.reranked_results,
                generated_answer=generated_answer
            )
        )

    async def start_verification(self, ev: WritingCompleteEvent):
        """Triggered by WritingCompleteEvent. Runs the VerifierAgent if enabled."""
        if not self.config.USE_VERIFIER:
            print("\n --- Verification skipped by configuration ---")
            self._set_status('Finalizing answer')
            self.context['timings']['total_workflow'] = time.monotonic() - self.context['workflow_start_time']
            self._set_status('Completed')
            await self.dispatch(StopEvent(
                result={
                    "final_answer": ev.generated_answer,
                    "verification_feedback": "Verification was disabled",
                    "timings": self.context['timings'],
                    "sources": self.context.get('source_refs', [])
                }
            ))
            return

        step_start_time = time.monotonic()
        self._set_status('Verifying answer')
        verification_result = await self.verifier_agent.averify_answer(
            query=ev.query,
            generated_answer=ev.generated_answer,
            source_context=ev.reranked_results
        )
        self.context['timings']['verification'] = self.context['timings'].get('verification', 0) + (time.monotonic() - step_start_time)

        is_faithful = verification_result.get("is_faithful", False)
        if is_faithful:
            print("\n--- Answer is faithful. Workflow complete. ---")
            self.context['timings']['total_workflow'] = time.monotonic() - self.context['workflow_start_time']
            self._set_status('Completed')
            await self.dispatch(StopEvent(
                result={
                    "final_answer": ev.generated_answer,
                    "verification_feedback": verification_result.get("feedback"),
                    "timings": self.context['timings'],
                    "sources": self.context.get('source_refs', [])
                }
            ))
            return

        self.context['rewrite_cycles'] += 1
        if self.context['rewrite_cycles'] >= self.config.MAX_REWRITES:
            print(f"\n--- Max rewrite limit ({self.config.MAX_REWRITES}) reached. Stopping. ---")
            self._set_status('Completed (verification failed)')
            await self.dispatch(StopEvent(
                result={
                    "final_answer": ev.generated_answer,
                    "verification_feedback": f"FINAL ATTEMPT FAILED: {verification_result.get('feedback')}",
                    "timings": self.context['timings'],
                    "sources": self.context.get('source_refs', [])
                }
            ))
            return

        print(f"\n--- Answer not faithful. Starting rewrite cycle {self.context['rewrite_cycles']}. ---")
        self._set_status('Applying feedback')
        await self.dispatch(
            RewriteEvent(
                query=ev.query,
                reranked_results=ev.reranked_results,
                feedback=verification_result.get("feedback"),
                previous_answer=ev.generated_answer
            )
        )


