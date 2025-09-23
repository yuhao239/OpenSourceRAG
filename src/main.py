# main.py
# This script serves as a command-line utility for managing the RAG system.
# Its primary function is to run the data ingestion workflow.

import sys
import asyncio
import nest_asyncio

from config import Config
from work_flows.ingestion import IngestionWorkflow
from events import StartIngestionEvent

# Apply nest_asyncio to allow running asyncio in environments like Jupyter or scripts
nest_asyncio.apply()

async def run_ingestion():
    """
    Initializes and runs the IngestionWorkflow.
    This function acts as the entry point for the entire ingestion process.
    """
    print("--- Initializing Ingestion Workflow ---")
    config = Config()
    
    # Create an instance of the workflow, setting a timeout for safety
    ingestion_workflow = IngestionWorkflow(config=config, timeout=600) # 10-minute timeout
    
    # Create the initial event that will trigger the first listener in the workflow
    initial_event = StartIngestionEvent()
    
    # Start the workflow and wait for it to complete
    result = await ingestion_workflow.run(initial_event)
    print(f"--- Ingestion Workflow execution has finished. ---")
    print(f"Result: {result}")

if __name__ == "__main__":
    """
    Main execution block to handle command-line arguments.
    """
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "ingest":
            print("Starting data ingestion process...")
            asyncio.run(run_ingestion())
            print("Ingestion complete.")
        # Placeholder for other potential utility commands in the future
        # elif command == "some_other_command":
        #     pass
        else:
            print(f"Unknown command: '{command}'. The primary command is 'ingest'.")
    else:
        print("Please provide a command. The primary command is 'ingest' to process documents.")
        print("Usage: python main.py ingest")

