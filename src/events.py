# Defines the structured event objects for our workflow system.

from dataclasses import dataclass, field
from typing import Any, List
from llama_index.core.schema import NodeWithScore

@dataclass
class StartIngestionEvent:
    """Event to trigger the start of the ingestion workflow."""
    pass

@dataclass
class IngestionCompleteEvent:
    """Event dispatched when the ingestion workflow is complete."""
    status: str
    num_documents_processed: int

@dataclass
class StopEvent:
    """A special event to signal the end of a workflow."""
    result: Any 

@dataclass 
class StartQueryEvent:
    """Event to trigger the start of the query workflow."""
    query: str

@dataclass 
class QueryPlanningCompleteEvent:
    """Event dispatched when the QueryPlannerAgent is finished."""
    query: str
    hyde_document: str
    requires_retrieval: bool
    result: dict = field(default_factory=dict)

@dataclass
class SearchCompleteEvent:
    """Event dispatched when the SearcherAgent is finished."""
    query: str
    search_results: List[NodeWithScore] = field(default_factory=list)

@dataclass 
class RerankCompleteEvent:
    """Event dispatched when the RerankerAgent is finished."""
    query: str
    reranked_results: List[NodeWithScore] = field(default=list)

@dataclass 
class WritingCompleteEvent:
    """Event dispatched by the WriterAgent, triggering the VerifierAgent."""
    query: str
    reranked_results: List[NodeWithScore]
    generated_answer: str 

@dataclass 
class RewriteEvent:
    """Event to trigger the WriterAgent to rewrite an answer based on feedback."""
    query: str
    reranked_results: List[NodeWithScore]
    feedback: str
    previous_answer: str
    # Structured verification details (optional)
    unsupported_sentences: List[str] = field(default_factory=list)
    missing_citations: List[str] = field(default_factory=list)
    required_facts: List[str] = field(default_factory=list)
