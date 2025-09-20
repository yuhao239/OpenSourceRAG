# test.py
"""
A standalone script to test the QueryWorkflow of the RAG system.

This script assumes:
1. The ingestion pipeline has already been run (`python main.py ingest`).
2. The ChromaDB database exists in the './db' directory.
3. The QueryWorkflow has been instrumented to return performance timings.
"""
import asyncio
import nest_asyncio
from work_flows.query_workflow import QueryWorkflow
from events import StartQueryEvent
from config import Config

# Apply nest_asyncio to allow running asyncio in environments like Jupyter or scripts
nest_asyncio.apply()

# --- Define Test Queries ---
# A list of questions designed to test various aspects of the RAG system.
# They are based on the content of the provided PDF documents.
TEST_QUERIES = [
    {
        "description": "Simple Fact Retrieval (BERT Paper)",
        "query": "What does BERT stand for?"
    },
    {
        "description": "Comparative Question (DeepSeek-R1 Paper)",
        "query": "How does DeepSeek-R1's performance on the AIME 2024 benchmark compare to OpenAI-01-1217?"
    },
    {
        "description": "Numerical/Detailed Question (Attention Paper)",
        "query": "In the 'Attention is All You Need' paper, what were the training costs for the big Transformer model?"
    },
    {
        "description": "Limitation Inquiry (DeepSeek-V3 Paper)",
        "query": "What are the limitations of the DeepSeek-V3 model regarding deployment?"
    },
    {
        "description": "Edge Case: Non-Retrieval / Conversational",
        "query": "Hello, how are you today?"
    },
    {
        "description": "Edge Case: Unanswerable from Context",
        "query": "What is the capital of Canada?"
    }
]

def print_report(query_data, result):
    """Formats and prints the results and analytics for a single query."""
    print(f"--- Test Case: {query_data['description']} ---")
    print(f"Query: {query_data['query']}\n")

    final_answer = result.get("final_answer", "N/A")
    feedback = result.get("verification_feedback", "N/A")
    timings = result.get("timings", {})

    print("✅ Final Answer:")
    print(f"{final_answer}\n")

    print("🕵️ Verification Feedback:")
    print(f"{feedback}\n")

    print("⏱️ Performance Analytics:")
    if timings:
        for module, duration in timings.items():
            print(f"  - {module.replace('_', ' ').title()}: {duration:.2f} seconds")
    else:
        print("  - No timing data available.")
    
    print("\n" + "="*80 + "\n")

async def main():
    """
    Initializes and runs the QueryWorkflow for all predefined test queries.
    """
    print("Initializing RAG Query Workflow for testing...")
    config = Config()
    query_workflow = QueryWorkflow(config=config, timeout=600) # 10-minute timeout for safety

    for query_data in TEST_QUERIES:
        try:
            # Create the initial event that will trigger the first listener in the workflow
            initial_event = StartQueryEvent(query=query_data["query"])
            
            # Start the workflow and wait for it to complete
            result = await query_workflow.run(initial_event)
            
            # Print a formatted report of the results
            print_report(query_data, result)

        except Exception as e:
            print(f"\n--- ERROR running test case: {query_data['description']} ---")
            print(f"Query: {query_data['query']}")
            print(f"Error: {e}\n")
            print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())