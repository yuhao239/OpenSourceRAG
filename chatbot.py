# chatbot.py
"""
The entry point for running the RAG system in a conversational,
stateful chatbot mode.

This script initializes the QueryWorkflow and manages the chat session,
maintaining conversation history in the workflow's context.
"""
import asyncio
import nest_asyncio

from work_flows.query_workflow import QueryWorkflow
from work_flows.ingestion import IngestionWorkflow
from events import StartQueryEvent, StartIngestionEvent
from config import Config
nest_asyncio.apply()

async def chat_session():
    """
    Manages a single, continuous chat session with the RAG agent.
    """
    print("--- Initializing Chatbot ---")
    print("Type 'exit' to end the conversation.")
    
    config = Config()
    
    # Initialize the workflow. The context will be managed within this instance.
    query_workflow = QueryWorkflow(config=config, timeout=600)
    query_workflow.context['chat_history'] = []

    while True:
        try:
            user_query = input("\nYou: ")
            if user_query.strip().lower() == '/ingest':
                    ingestion_workflow = IngestionWorkflow(config=config, timeout=120)
                    initial_event = StartIngestionEvent()
                    await ingestion_workflow.run(initial_event)
                    print("--- Restarted query engine with updated knowledge base. ---")
                    continue
            
            if user_query.strip().lower() == 'exit':
                print("--- Chat Session Ended ---")
                break

            # Initialize StartEvent
            initial_event = StartQueryEvent(query=user_query)
            
            # Run the workflow for the current query
            result = await query_workflow.run(initial_event)
            final_answer = result.get("final_answer", "Sorry, I encountered an error.")

            print(f"\nAssistant: {final_answer}")

            # Update the chat history within the workflow's context for the next turn
            query_workflow.context['chat_history'].append({"role": "user", "content": user_query})
            query_workflow.context['chat_history'].append({"role": "assistant", "content": final_answer})

        except (KeyboardInterrupt, EOFError):
            print("\n--- Chat Session Interrupted ---")
            break
        except Exception as e:
            print(f"\n--- An error occurred: {e} ---")
            print("Restarting the session...")
            # Optionally re-initialize the workflow for a clean state
            query_workflow = QueryWorkflow(config=config, timeout=600)
            query_workflow.context['chat_history'] = []


if __name__ == "__main__":
    asyncio.run(chat_session())
