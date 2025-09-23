# work_flows/ingestion.py
# This file defines the IngestionWorkflow which orchestrates the ingestion process.

from llama_index.core import SimpleDirectoryReader
from .base import Workflow
from events import StartIngestionEvent, IngestionCompleteEvent, StopEvent
from agents.ingestion_agent import IngestionAgent
from config import Config

class IngestionWorkflow(Workflow):
    """
    Orchestrates the offline process of loading, chunking, embedding,
    and storing documents in the vector database.
    """
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.ingestion_agent = IngestionAgent(config)
        self.add_listener(StartIngestionEvent, self.start_ingestion)

    async def start_ingestion(self, event: StartIngestionEvent):
        """
        The entry point for the ingestion workflow, triggered by StartIngestionEvent.
        """
        print("\n--- Ingestion Workflow Started ---")
        
        # Load Documents
        print(f"Loading documents from '{self.config.DATA_DIR}'...")
        reader = SimpleDirectoryReader(self.config.DATA_DIR)
        documents = reader.load_data()
        
        if not documents:
            print(f"No documents found in '{self.config.DATA_DIR}'.")
            await self.dispatch(IngestionCompleteEvent(status="No documents found", num_documents_processed=0))
            await self.dispatch(StopEvent(result="Finished: No documents."))
            return
            
        print(f"Loaded {len(documents)} document(s).")

        # Run the Ingestion Agent
        num_nodes = await self.ingestion_agent.process_documents(documents)

        # Dispatch completion event and stop the workflow
        await self.dispatch(IngestionCompleteEvent(status="Success", num_documents_processed=num_nodes))
        await self.dispatch(StopEvent(result=f"Successfully processed {num_nodes} nodes."))
        print("--- Ingestion Workflow Finished ---")

