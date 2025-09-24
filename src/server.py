# server.py
"""
This script creates a FastAPI web server to expose the RAG chatbot's
functionality through a RESTful API.

This separates the core RAG and persistence logic from the user interface,
allowing any frontend application to interact with the chatbot.
"""
import uvicorn
import chromadb
import asyncio
import os
from fastapi import (
    FastAPI, 
    HTTPException,
    UploadFile,
    File,
    BackgroundTasks
)
import shutil
from pydantic import BaseModel
from contextlib import asynccontextmanager

from config import Config
from database import DatabaseManager
from work_flows.query_workflow import QueryWorkflow
from work_flows.ingestion import IngestionWorkflow
from events import StartIngestionEvent
from events import StartQueryEvent
from typing import List

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

@app.get("/documents", response_model=List[str])
async def get_documents():
    """Lists all the files in the data directory."""
    if not os.path.exists(app_config.DATA_DIR):
        return []
    return os.listdir(app_config.DATA_DIR)

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a new PDF file to the data directory."""
    if not os.path.exists(app_config.DATA_DIR):
        os.makedirs(app_config.DATA_DIR)
    
    filepath = os.path.join(app_config.DATA_DIR, file.filename)

    with open(filepath, 'wb') as buffer:
        buffer.write(await file.read())
    
    return {"filename": file.filename, "status": "success"}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Deletes a specific file from the data directory."""
    filepath = os.path.join(app_config.DATA_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found.")

    os.remove(filepath)
    return {"filename": filename, "status": "deleted"}

def run_ingestion_task():
    """The actual ingestion logic to be run in the background."""
    print("--- Starting ingestion in the background. ---")
    # 1. Safely clear the old ChromaDB collection through the client
    try:
        client = chromadb.PersistentClient(path=app_config.CHROMA_PERSIST_DIR)
        # Check if the collection exists before trying to delete it
        collections = client.list_collections()
        if any(c.name == app_config.CHROMA_COLLECTION_NAME for c in collections):
            print(f"Deleting existing ChromaDB collection: '{app_config.CHROMA_COLLECTION_NAME}'")
            client.delete_collection(name=app_config.CHROMA_COLLECTION_NAME)
            print("Cleared old ChromaDB collection.")
    except Exception as e:
        print(f"Error clearing ChromaDB collection: {e}")

    # 2. Remove the old nodes file 
    if os.path.exists(app_config.NODES_PATH):
        os.remove(app_config.NODES_PATH)
        print("Removed old nodes.pkl file")
    
    ingestion_workflow = IngestionWorkflow(config=app_config, timeout=300)
    initial_event = StartIngestionEvent()

    asyncio.run(ingestion_workflow.run(initial_event))
    print("--- Background ingestion task finished. ---")

@app.post("/ingest")
async def trigger_ingestion(background_tasks: BackgroundTasks):
    """Triggers the data ingestion workflow in the background."""
    background_tasks.add_task(run_ingestion_task)
    return {"message": "Data ingestion has been started in the background. It may take a few moments to complete."}

# --- How to run the server ---
if __name__ == "__main__":
    # To run this server:
    # 1. Ensure your PostgreSQL container is running.
    # 2. Open your terminal in this directory.
    # 3. Run the command: uvicorn server:app --reload
    print("To run the API server, use the following command:")
    print("uvicorn server:app --reload")

