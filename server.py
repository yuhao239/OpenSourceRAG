# server.py
"""
This script creates a FastAPI web server to expose the RAG chatbot's
functionality through a RESTful API.

This separates the core RAG and persistence logic from the user interface,
allowing any frontend application to interact with the chatbot.
"""
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from config import Config
from database import DatabaseManager
from work_flows.query_workflow import QueryWorkflow
from events import StartQueryEvent

# --- Data Models for API requests and responses ---
class QueryRequest(BaseModel):
    """Defines the structure for a user's query POST request."""
    query: str

class ConversationRequest(BaseModel):
    """Defines the structure for a new conversation request."""
    title: str

# --- Global objects ---
# These objects are initialized once and shared across all API requests.
app_config = Config()
db_manager = DatabaseManager(dsn=app_config.db_connection_string)

# --- FastAPI Lifespan Events ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's startup and shutdown events.
    This is the recommended way to handle resources like database connections.
    """
    print("--- Starting up server and connecting to database... ---")
    await db_manager.connect()
    await db_manager.init_db() # Ensure tables exist
    yield
    print("--- Shutting down server and closing database connection... ---")
    await db_manager.close()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Multi-Agent RAG Chatbot API",
    description="An API for interacting with the advanced RAG chatbot.",
    version="1.0.0",
    lifespan=lifespan
)

# --- API Endpoints ---
@app.get("/conversations", summary="List all conversations")
async def get_all_conversations():
    """Retrieves a list of all past conversations."""
    conversations = await db_manager.get_conversations()
    return {"conversations": conversations}

@app.post("/conversations", summary="Create a new conversation")
async def create_new_conversation(request: ConversationRequest):
    """Creates a new, empty conversation and returns its ID and title."""
    conv_id = await db_manager.create_conversation(title=request.title)
    return {"conversation_id": conv_id, "title": request.title}

@app.get("/conversations/{conversation_id}", summary="Get messages for a conversation")
async def get_conversation_messages(conversation_id: int):
    """Retrieves the full message history for a specific conversation."""
    history = await db_manager.get_messages(conversation_id)
    # The get_messages function in database.py returns a list. An empty list is a valid history.
    if history is None:
        raise HTTPException(status_code=404, detail="Conversation not found or failed to retrieve messages.")
    return {"messages": history}

@app.get("/health")
async def health_check():
    """A simple endpoint to confirm the API is running."""
    return {"status": "ok"}

@app.post("/conversations/{conversation_id}/query", summary="Post a query to a conversation")
async def post_query(conversation_id: int, request: QueryRequest):
    """
    The main endpoint for interacting with the RAG agent.
    1. Loads the conversation history.
    2. Runs the full RAG workflow.
    3. Saves the new messages to the database.
    4. Returns the agent's final answer.
    """
    # 1. Load conversation history
    chat_history = await db_manager.get_messages(conversation_id)
    if chat_history is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # --- FIX: Initialize the workflow here, just-in-time ---
    query_workflow = QueryWorkflow(config=app_config, timeout=600)
    query_workflow.context['chat_history'] = chat_history
    
    # We manually add the user's new message for the planner agent's context
    query_workflow.context['chat_history'].append({"role": "user", "content": request.query})

    # 2. Run the RAG workflow
    initial_event = StartQueryEvent(query=request.query)
    result = await query_workflow.run(initial_event)
    final_answer = result.get("final_answer", "Sorry, an error occurred.")

    # 3. Save the user and assistant messages to the database
    await db_manager.add_message(conversation_id, 'user', request.query)
    await db_manager.add_message(conversation_id, 'assistant', final_answer, metadata=result)

    # 4. Return the final answer
    return {"response": final_answer, "metadata": result}

# --- How to run the server ---
if __name__ == "__main__":
    # To run this server:
    # 1. Ensure your PostgreSQL container is running.
    # 2. Open your terminal in this directory.
    # 3. Run the command: uvicorn server:app --reload
    print("To run the API server, use the following command:")
    print("uvicorn server:app --reload")

