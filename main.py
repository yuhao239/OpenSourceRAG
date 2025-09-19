# main.py
# This is the entry point for running our agent workflow.
# Its role is to initialize and trigger the correct workflow.
import os, sys
import asyncio
import nest_asyncio

from config import Config
# Correctly import from the 'workflows' package
from work_flows.ingestion import IngestionWorkflow
from work_flows.query_workflow import QueryWorkflow
# Correctly import from the 'events' module
from events import StartIngestionEvent, StartQueryEvent

# Apply nest_asyncio to allow running asyncio in environments like Jupyter
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
    await ingestion_workflow.run(initial_event)
    print("--- Ingestion Workflow execution has finished. ---")

async def run_query(query: str):
    """Initializes and runs the QueryWorkflow."""
    print("--- Initializing Query Workflow ---")
    config = Config()
    query_workflow = QueryWorkflow(config=config, timeout=300)
    initial_event = StartQueryEvent(query=query)
    await query_workflow.run(initial_event)
    print("--- Query Workflow execution has finished. ---")


# This is the main execution block of the script.
if __name__ == "__main__":
    # This script is currently set up to run the ingestion workflow.
    # In the future, we will add logic here to choose between different
    # workflows, such as ingestion or querying.
        if len(sys.argv) > 1:
            command = sys.argv[1]
            if command == "ingest":
                asyncio.run(run_ingestion())
            elif command == "query":
                if len(sys.argv) > 2:
                    user_query = " ".join(sys.argv[2:])
                    asyncio.run(run_query(user_query))
                else:
                    print("Please provide a query. Usage: python main.py query <your question here>")
            else:
                print(f"Unknown command: '{command}'. Use 'ingest' or 'query'.")
        else:
            print("Please provide a command. Use 'ingest' or 'query'.")

