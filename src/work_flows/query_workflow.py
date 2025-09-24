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

    async def start_query_planning(self, event: StartQueryEvent):
        """Entry point for the query workflow."""
        print("\n--- Query Workflow Started ---")
        self._set_status('Planning query')

        self.context['rewrite_cycles'] = 0
        self.context['timings'] = {}
        chat_history = self.context.get('chat_history', [])
        self.context['workflow_start_time'] = time.monotonic()
        step_start_time = time.monotonic()

        plan = await self.query_planner_agent.aplan_query(event.query, chat_history=chat_history)
        self.context['timings']['query_planning'] = time.monotonic() - step_start_time

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
            self._set_status('Completed')
            await self.dispatch(StopEvent(result={
                "final_answer": final_answer,
                "verification_feedback": "N/A (Direct Answer)",
                'timings': self.context['timings']
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
                    "timings": self.context['timings']
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
                    "timings": self.context['timings']
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
                    "timings": self.context['timings']
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

