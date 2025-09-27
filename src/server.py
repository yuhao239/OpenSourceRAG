# server.py
"""
FastAPI web server exposing the RAG chatbot API.
"""
import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional
from uuid import uuid4

import chromadb
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import Config
from database import DatabaseManager
from events import StartIngestionEvent, StartQueryEvent
from work_flows.ingestion import IngestionWorkflow
from work_flows.query_workflow import QueryWorkflow


class QueryRequest(BaseModel):
    query: str


class ConversationRequest(BaseModel):
    title: str


app_config = Config()
db_manager = DatabaseManager(dsn=app_config.db_connection_string)
ingestion_lock: Optional[asyncio.Lock] = None
_current_ingestion_task: Optional[asyncio.Task] = None
query_jobs: dict[str, dict] = {}


def _is_ingestion_running() -> bool:
    return _current_ingestion_task is not None and not _current_ingestion_task.done()


async def _run_ingestion_workflow():
    global ingestion_lock, _current_ingestion_task

    if ingestion_lock is None:
        ingestion_lock = asyncio.Lock()

    async with ingestion_lock:
        print("--- Starting ingestion job ---")

        try:
            client = chromadb.PersistentClient(path=app_config.CHROMA_PERSIST_DIR)
            collections = client.list_collections()
            if any(c.name == app_config.CHROMA_COLLECTION_NAME for c in collections):
                print(f"Deleting existing ChromaDB collection: '{app_config.CHROMA_COLLECTION_NAME}'")
                client.delete_collection(name=app_config.CHROMA_COLLECTION_NAME)
                print("Cleared old ChromaDB collection.")
        except Exception as exc:
            print(f"Error clearing ChromaDB collection: {exc}")

        if os.path.exists(app_config.NODES_PATH):
            try:
                os.remove(app_config.NODES_PATH)
                print("Removed old nodes.pkl file")
            except OSError as exc:
                print(f"Failed to remove nodes.pkl: {exc}")

        ingestion_workflow = IngestionWorkflow(config=app_config, timeout=300)
        await ingestion_workflow.run(StartIngestionEvent())
        print("--- Ingestion job finished ---")

    _current_ingestion_task = None


def schedule_ingestion() -> bool:
    global _current_ingestion_task

    if _is_ingestion_running():
        print("Ingestion request ignored because a job is already running.")
        return False

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        print("Failed to obtain running loop; ingestion not scheduled.")
        return False

    _current_ingestion_task = loop.create_task(_run_ingestion_workflow())
    return True


@asynccontextmanager
async def lifespan(_: FastAPI):
    global ingestion_lock

    print("--- Starting up server and connecting to database... ---")
    ingestion_lock = asyncio.Lock()
    await db_manager.connect()
    await db_manager.init_db()
    yield
    print("--- Shutting down server and closing database connection... ---")
    await db_manager.close()


app = FastAPI(
    title="Multi-Agent RAG Chatbot API",
    description="REST API for the multi-agent RAG chatbot.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/conversations", summary="List conversations")
async def get_all_conversations():
    conversations = await db_manager.get_conversations()
    return {"conversations": conversations}


@app.post("/conversations", summary="Create conversation")
async def create_new_conversation(request: ConversationRequest):
    conv_id = await db_manager.create_conversation(title=request.title)
    return {"conversation_id": conv_id, "title": request.title}


@app.get("/conversations/{conversation_id}", summary="Get conversation messages")
async def get_conversation_messages(conversation_id: int):
    history = await db_manager.get_messages(conversation_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Conversation not found or failed to retrieve messages.")
    return {"messages": history}


@app.get("/health")
async def health_check():
    return {"status": "ok", "ingestion_running": _is_ingestion_running()}


@app.post("/conversations/{conversation_id}/query")
async def post_query(conversation_id: int, request: QueryRequest):
    chat_history = await db_manager.get_messages(conversation_id)
    if chat_history is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    job_id = str(uuid4())
    job_record = {
        "job_id": job_id,
        "conversation_id": conversation_id,
        "status": "running",
        "phase": "Starting",
        "created_at": time.time(),
    }
    query_jobs[job_id] = job_record

    async def run_query_job():
        local_history = list(chat_history)
        initial_event = StartQueryEvent(query=request.query)

        def update_phase(phase: str):
            job_record["phase"] = phase
            job_record["updated_at"] = time.time()

        try:
            query_workflow = QueryWorkflow(config=app_config, timeout=600)
            query_workflow.context['chat_history'] = local_history
            query_workflow.context['set_status'] = update_phase
            local_history.append({"role": "user", "content": request.query})

            await db_manager.add_message(conversation_id, 'user', request.query)

            result = await query_workflow.run(initial_event)
            if result is None:
                raise RuntimeError("Workflow returned no result.")

            if not isinstance(result, dict):
                raise RuntimeError(f"Workflow returned unexpected payload of type {type(result).__name__}")

            if result.get("error"):
                phase = result.get("phase", "query planning")
                raise RuntimeError(f"Workflow failed during {phase}: {result['error']}")

            final_answer = result.get("final_answer") or "Sorry, an error occurred."

            await db_manager.add_message(conversation_id, 'assistant', final_answer, metadata=result)

            update_phase("Completed")
            job_record["status"] = "completed"
            sources = result.get("sources") or []
            job_record["result"] = {
                "response": final_answer,
                "metadata": result,
                "sources": sources,
            }

        except Exception as exc:
            job_record["status"] = "failed"
            job_record["error"] = str(exc)
            job_record["phase"] = "Failed"
            print(f"Query job {job_id} failed: {exc}")
        finally:
            job_record["finished_at"] = time.time()

    asyncio.create_task(run_query_job())
    return {"job_id": job_id, "status": "running"}


@app.get("/query-jobs/{job_id}")
async def get_query_job(job_id: str):
    job = query_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.copy()


@app.get("/documents", response_model=List[str])
async def get_documents():
    if not os.path.exists(app_config.DATA_DIR):
        return []
    return sorted(os.listdir(app_config.DATA_DIR))


def _validate_filename(filename: str) -> str:
    safe_name = os.path.basename(filename)
    if not safe_name or safe_name.startswith("..") or safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename provided.")
    return safe_name


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    filename = _validate_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(app_config.DATA_DIR, exist_ok=True)
    filepath = os.path.join(app_config.DATA_DIR, filename)

    try:
        with open(filepath, 'wb') as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)
    except Exception as exc:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.") from exc
    finally:
        await file.close()

    ingestion_started = schedule_ingestion()
    return {"filename": filename, "status": "success", "ingestion_started": ingestion_started}


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    safe_name = _validate_filename(filename)
    filepath = os.path.join(app_config.DATA_DIR, safe_name)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found.")
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=400, detail="Requested path is not a file.")

    os.remove(filepath)
    ingestion_started = schedule_ingestion()
    return {"filename": safe_name, "status": "deleted", "ingestion_started": ingestion_started}




@app.get("/documents/view/{filename}")
async def view_document(filename: str, request: Request):
    """Streams a document for preview in the UI."""
    safe_name = _validate_filename(filename)
    filepath = os.path.join(app_config.DATA_DIR, safe_name)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found.")

    allowed_origins = getattr(request.app.state, "allowed_origins", ["*"])
    allow_credentials = getattr(request.app.state, "allow_credentials", False)
    origin = request.headers.get("origin")
    allow_origin = "*"
    if allowed_origins and "*" not in allowed_origins:
        if origin in allowed_origins:
            allow_origin = origin
        else:
            allow_origin = allowed_origins[0]

    headers = {
        "Content-Disposition": f'inline; filename="{safe_name}"',
        "Access-Control-Allow-Origin": allow_origin,
        "Access-Control-Allow-Headers": "Range, Content-Type, Accept",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Expose-Headers": "Accept-Ranges, Content-Length, Content-Range",
        "X-Frame-Options": "ALLOWALL",
        "Content-Security-Policy": "frame-ancestors 'self' http://localhost:8501 http://127.0.0.1:8501",
        "Cross-Origin-Resource-Policy": "cross-origin",
        "Accept-Ranges": "bytes",
    }
    if allow_credentials and allow_origin != "*":
        headers["Access-Control-Allow-Credentials"] = "true"
    if allow_origin != "*":
        headers["Vary"] = "Origin"

    return FileResponse(filepath, media_type="application/pdf", headers=headers)

@app.post("/ingest")
async def trigger_ingestion():
    started = schedule_ingestion()
    if started:
        message = "Data ingestion has been started in the background. It may take a few moments to complete."
    else:
        message = "Data ingestion is already running. New request ignored."
    return {"message": message, "ingestion_running": _is_ingestion_running()}


@app.get("/pdfjs/viewer", response_class=HTMLResponse)
async def pdfjs_viewer_page():
    """Serves a lightweight PDF.js viewer page that accepts query params:
    - file: absolute or API-relative URL to the PDF
    - page: 1-based page number
    - search: initial search query to highlight
    The page parses params on the client side.
    """
    viewer_path = os.path.join(os.path.dirname(__file__), "static", "pdfjs_viewer.html")
    if not os.path.exists(viewer_path):
        raise HTTPException(status_code=500, detail="PDF.js viewer asset not found.")
    return FileResponse(viewer_path, media_type="text/html")


configured_origins = getattr(app_config, "CORS_ALLOW_ORIGINS", ["*"]) or ["*"]
if any(origin == "*" for origin in configured_origins):
    allow_origins = ["*"]
    allow_credentials = False
else:
    allow_origins = configured_origins
    allow_credentials = True

app.state.allowed_origins = allow_origins
app.state.allow_credentials = allow_credentials

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Accept-Ranges", "Content-Length", "Content-Range"],
)

if __name__ == "__main__":
    print("To run the API server, use the following command:")
    print("uvicorn server:app --reload")

