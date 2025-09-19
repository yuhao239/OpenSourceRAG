# workflows/query_workflow.py
# Workflow orchestrating the whole query to answer pipeline

from .base import Workflow
from events import (
    StartQueryEvent,
    QueryPlanningCompleteEvent,
    StopEvent,
    SearchCompleteEvent,
    RerankCompleteEvent
)
from agents.query_planner_agent import QueryPlannerAgent
from config import Config
from agents.searcher_agent import SearcherAgent
from agents.reranker_agent import RerankerAgent

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
        
        # Register the listener for the starting event
        self.add_listener(StartQueryEvent, self.start_query_planning)
        self.add_listener(QueryPlanningCompleteEvent, self.start_search)
        self.add_listener(SearchCompleteEvent, self.start_reranking)

    async def start_query_planning(self, event: StartQueryEvent):
        """
        The entry point for the query workflow, triggered by StartQueryEvent.
        """
        print("\n--- Query Workflow Started ---")
        
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
    
        # Stop the workflow for now.
        await self.dispatch(StopEvent(result=reranked_results))
        print("--- Search finished. ---")