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

class QueryWorkflow(Workflow):
    """
    Orchestrates the real-time, interactive query process.
    """
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.query_planner_agent = QueryPlannerAgent(config) # LLM not specified, needs further testing 
        self.searcher_agent = SearcherAgent(config)
        self.reranker_agent = RerankerAgent(config)
        self.writer_agent = WriterAgent(config)
        self.verifier_agent = VerifierAgent(config)
        
        # Register the listener for the starting event
        self.add_listener(StartQueryEvent, self.start_query_planning)
        self.add_listener(QueryPlanningCompleteEvent, self.start_search)
        self.add_listener(SearchCompleteEvent, self.start_reranking)

        # The WriterAgent can be triggered by either event
        self.add_listener(RerankCompleteEvent, self.handle_writing_request)
        self.add_listener(RewriteEvent, self.handle_writing_request)

        self.add_listener(WritingCompleteEvent, self.start_verification)

    async def start_query_planning(self, event: StartQueryEvent):
        """
        The entry point for the query workflow, triggered by StartQueryEvent.
        """
        print("\n--- Query Workflow Started ---")

        # Initialize the rewrite cycles attribute for VerifierAgent
        self.context['rewrite_cycles'] = 0         
        
        # Run the Query Planner Agent
        plan = await self.query_planner_agent.aplan_query(event.query)

        # Dispatch completion event to trigger the next agent (Searcher)
        await self.dispatch(
            QueryPlanningCompleteEvent(
                query=plan.get("query"),
                hyde_document=plan.get("hyde_document"),
                requires_retrieval=plan.get("requires_retrieval", True),
                result=plan
            )
        )
    
    async def start_search(self, ev: QueryPlanningCompleteEvent):
        """
        Triggered by QueryPlanningCompleteEvent. Runs the SearcherAgent. 
        """
        if not ev.requires_retrieval:
            print("\n--- Retrieval not required. Skipping Search. ---")

            # dispatch to write agent 
            # for now halt the flow of operations
            await self.dispatch(StopEvent(result={"final_answer": "Retrieval was not required for this query."}))
            return 
        
        search_results = await self.searcher_agent.asearch(
            query=ev.query,
            hyde_document=ev.hyde_document
        )

        # Dispatch completion event to trigger next agent (reranker)
        await self.dispatch(
            SearchCompleteEvent(
                query=ev.query,
                search_results=search_results
            )
        )
    
    async def start_reranking(self, ev: SearchCompleteEvent):
        """
        Triggered by SearchCompleteEvent. Runs the RerankerAgent.
        """

        reranked_results = await self.reranker_agent.arerank(
            query=ev.query,
            documents=ev.search_results
        )

        # Dispatch completion event to pass off to final WriterAgent
        await self.dispatch(
            RerankCompleteEvent(
                query=ev.query,
                reranked_results=reranked_results
            )
        )


    async def start_verification(self, ev: WritingCompleteEvent):
        """
        Triggered by WritingCompleteEvent. Runs the VerifierAgent if enabled. 
        This is the final step before stopping the workflow.
        """

        if not self.config.USE_VERIFIER:
            print("\n --- Verification skipped by configuration ---")
            await self.dispatch(StopEvent(
                    result = {
                        "final_answer": ev.generated_answer,
                        "verification_feedback": "Verification was disabled"
                        }
                ))
            return 

        verification_result = await self.verifier_agent.averify_answer(
            query=ev.query,
            generated_answer=ev.generated_answer,
            source_context=ev.reranked_results
        )

        
        is_faithful = verification_result.get("is_faithful", False)
        if is_faithful:
            # Answer is good, stop the workflow
            print("\n--- Answer is faithful. Workflow complete. ---")
            await self.dispatch(StopEvent(
            result={
                "final_answer": ev.generated_answer,
                "verification_feedback": verification_result.get("feedback")
                }
            ))
            
        else:
            # Answer is not faithful, check rewrite limit
            self.context['rewrite_cycles'] += 1 
            if self.context['rewrite_cycles'] >= self.config.MAX_REWRITES:
                 print(f"\n--- Max rewrite limit ({self.config.MAX_REWRITES}) reached. Stopping. ---")
                 await self.dispatch(StopEvent(
                        result={
                            "final_answer": ev.generated_answer,
                            "verification_feedback": f"FINAL ATTEMPT FAILED: {verification_result.get('feedback')}"
                            }
                ))
            else:
                # Limit not reached, dispatch RewriteEvent
                print(f"\n--- Answer not faithful. Starting rewrite cycle {self.context['rewrite_cycles']}. ---")
                await self.dispatch(
                    RewriteEvent(
                        query=ev.query,
                        reranked_results=ev.reranked_results,
                        feedback=verification_result.get("feedback"),
                        previous_answer=ev.generated_answer
                    )
                )
    
    async def handle_writing_request(self, ev: RerankCompleteEvent | RewriteEvent):
        """
        A single handler for both initial writing and subsequent rewrites.
        Can be triggered by either a RerankCompleteEvent or a RewriteEvent. 
        """
        feedback = ev.feedback if isinstance(ev, RewriteEvent) else None
        previous_answer = ev.previous_answer if isinstance(ev, RewriteEvent) else None
        
        generated_answer = await self.writer_agent.awrite_answer(
            query=ev.query,
            reranked_results=ev.reranked_results,
            feedback=feedback,
            previous_answer=previous_answer
        )

        # Dispatch WritingCompleteEvent to trigger the VerifierAgent
        await self.dispatch(
            WritingCompleteEvent(
                query=ev.query,
                reranked_results=ev.reranked_results,
                generated_answer=generated_answer
            )
        )