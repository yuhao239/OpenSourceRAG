# ui.py
"""
Streamlit UI for the Multi-Agent RAG Chatbot.
"""
import json
import os
import time
from typing import Any, Dict, List
from urllib.parse import quote

import requests
import streamlit as st
import streamlit.components.v1 as components

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
PUBLIC_API_BASE_URL = os.environ.get("PUBLIC_API_BASE_URL", API_BASE_URL)
POLL_INTERVAL_SECONDS = float(os.environ.get("INGESTION_POLL_INTERVAL", "3"))
QUERY_POLL_INTERVAL_SECONDS = float(os.environ.get("QUERY_POLL_INTERVAL", "0.7"))
QUERY_POLL_TIMEOUT = float(os.environ.get("QUERY_POLL_TIMEOUT", "600"))


def _safe_request(method: str, endpoint: str, **kwargs) -> requests.Response | None:
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.request(method, url, timeout=30, **kwargs)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as exc:
        st.error(f"Request to {endpoint} failed: {exc}")
        return None


def get_conversations() -> List[Dict[str, Any]]:
    response = _safe_request("GET", "/conversations")
    if not response:
        return []
    return response.json().get("conversations", [])


def create_conversation(title: str) -> Dict[str, Any] | None:
    response = _safe_request("POST", "/conversations", json={"title": title})
    if not response:
        return None
    return response.json()


def _process_message_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata = payload.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
    sources = payload.get("sources") or metadata.get("sources") or []
    return {
        "role": payload.get("role"),
        "content": payload.get("content"),
        "metadata": metadata,
        "sources": sources,
    }


def get_messages(conversation_id: int) -> List[Dict[str, Any]]:
    response = _safe_request("GET", f"/conversations/{conversation_id}")
    if not response:
        return []
    messages = response.json().get("messages", [])
    return [_process_message_payload(msg) for msg in messages]


def start_query_job(conversation_id: int, query: str) -> Dict[str, Any] | None:
    response = _safe_request("POST", f"/conversations/{conversation_id}/query", json={"query": query})
    if not response:
        return None
    return response.json()


def get_query_job(job_id: str) -> Dict[str, Any] | None:
    response = _safe_request("GET", f"/query-jobs/{job_id}")
    if not response:
        return None
    return response.json()


def get_documents() -> List[str]:
    response = _safe_request("GET", "/documents")
    if not response:
        return []
    return response.json()


def delete_document(filename: str) -> bool:
    response = _safe_request("DELETE", f"/documents/{filename}")
    if not response:
        return False
    ingestion_started = response.json().get("ingestion_started", False)
    if ingestion_started:
        st.info("Ingestion started to refresh the knowledge base.")
    return True


def trigger_ingestion() -> Dict[str, Any] | None:
    response = _safe_request("POST", "/ingest")
    if not response:
        return None
    return response.json()


def get_ingestion_status() -> bool:
    response = _safe_request("GET", "/health")
    if not response:
        return False
    return bool(response.json().get("ingestion_running", False))


def format_timestamp(ts: float | None) -> str:
    if not ts:
        return ""
    return time.strftime("%H:%M:%S", time.localtime(ts))


def build_viewer_url(source: Dict[str, Any]) -> str:
    """Build URL for the internal PDF.js viewer with page + search."""
    file_name = source.get("file")
    if not file_name:
        return ""
    pdf_endpoint = f"{PUBLIC_API_BASE_URL}/documents/view/{quote(file_name)}"
    page_number = source.get("page")
    highlight_keyword = source.get("highlight_keyword")

    from urllib.parse import urlencode, quote as _quote
    qs = {"file": pdf_endpoint}
    if page_number:
        try:
            qs["page"] = int(page_number)
        except Exception:
            pass
    if highlight_keyword:
        qs["search"] = highlight_keyword
    return f"{PUBLIC_API_BASE_URL}/pdfjs/viewer?{urlencode(qs, quote_via=_quote)}"



def render_pdf_preview(source: Dict[str, Any]):
    st.divider()
    title = source.get("file") or "Selected Source"
    st.subheader(f"Source preview - {title}")
    pdf_url = build_viewer_url(source)
    if not pdf_url:
        st.info("Preview unavailable for this source.")
        return

    viewer_src = pdf_url

    components.iframe(
        viewer_src,
        height=740,
    )
def render_sources(prefix: str, sources: List[Dict[str, Any]]):
    if not sources:
        return
    st.caption("_Sources_")
    for idx, src in enumerate(sources, start=1):
        label = src.get("file") or "Unknown source"
        page_label = src.get("page_label")
        page_number = src.get("page")
        if page_label:
            label += f" ({page_label})"
        elif page_number is not None:
            label += f" (p.{page_number})"

        excerpt = src.get("excerpt")
        cols = st.columns([0.8, 0.2])
        with cols[0]:
            st.markdown(f"**{idx}.** {label}")
            if excerpt:
                st.caption(excerpt)
        with cols[1]:
            if st.button("Open", key=f"{prefix}_{idx}"):
                st.session_state.selected_source = src


st.set_page_config(page_title="Multi-Agent RAG Chatbot", layout="wide")
st.title("Multi-Agent RAG Chatbot")

if "selected_conversation_id" not in st.session_state:
    st.session_state.selected_conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingestion_running" not in st.session_state:
    st.session_state.ingestion_running = False
if "ingestion_was_running" not in st.session_state:
    st.session_state.ingestion_was_running = False
if "ingestion_completed_at" not in st.session_state:
    st.session_state.ingestion_completed_at = None
if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None
if "selected_source" not in st.session_state:
    st.session_state.selected_source = None

with st.sidebar:
    st.header("Conversations")
    if st.button("New Chat"):
        new_conv_title = f"New Chat {int(time.time())}"
        new_conv = create_conversation(new_conv_title)
        if new_conv:
            st.session_state.selected_conversation_id = new_conv["conversation_id"]
            st.session_state.messages = []
            st.session_state.selected_source = None
            st.rerun()

    st.subheader("Past Chats")
    conversations = get_conversations()
    if not conversations:
        st.caption("No past conversations found.")
    else:
        for conv in conversations:
            if st.button(f"{conv['title']} (ID: {conv['id']})", key=f"conv_{conv['id']}"):
                st.session_state.selected_conversation_id = conv["id"]
                st.session_state.messages = get_messages(conv["id"])
                st.session_state.selected_source = None
                st.rerun()

with st.sidebar:
    st.header("Knowledge Base")
    st.caption("UI build hash: 89164eb4169f2f729a98451bdc77becc49323479")

    ingestion_running = get_ingestion_status()
    st.session_state.ingestion_running = ingestion_running

    status_label = "Running" if ingestion_running else "Idle"
    st.caption(f"Ingestion status: {status_label}")

    if ingestion_running:
        st.session_state.ingestion_was_running = True
        st.caption(f"Auto-refreshing every {int(POLL_INTERVAL_SECONDS)}s while ingestion runs.")
    else:
        if st.session_state.ingestion_was_running:
            st.session_state.ingestion_was_running = False
            st.session_state.ingestion_completed_at = time.time()
            st.toast("Knowledge base refreshed.")
        if st.session_state.ingestion_completed_at:
            last_refresh = format_timestamp(st.session_state.ingestion_completed_at)
            if last_refresh:
                st.caption(f"Last refreshed at {last_refresh}.")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_filename:
        with st.spinner("Uploading..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = _safe_request("POST", "/documents/upload", files=files)
            if response:
                st.session_state.last_uploaded_filename = uploaded_file.name
                payload = response.json()
                st.success(f"File '{uploaded_file.name}' uploaded successfully.")
                if payload.get("ingestion_started"):
                    st.info("Ingestion started to include the new document.")
                st.rerun()
            else:
                st.session_state.last_uploaded_filename = None
                st.error("Upload failed.")

    st.subheader("Managed Files")
    doc_list = get_documents()
    if not doc_list:
        st.caption("No documents found.")
    else:
        for doc_name in doc_list:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(doc_name)
            with col2:
                if st.button("x", key=f"del_{doc_name}"):
                    if delete_document(doc_name):
                        st.success(f"Deleted '{doc_name}'.")
                        st.rerun()

    st.divider()
    if st.button("Update Knowledge Base", disabled=ingestion_running):
        with st.spinner("Starting ingestion..."):
            result = trigger_ingestion()
            if result:
                st.session_state.ingestion_running = result.get("ingestion_running", False)
                if st.session_state.ingestion_running:
                    st.session_state.ingestion_was_running = True
                st.success(result.get("message", "Ingestion triggered."))
            else:
                st.error("Failed to start ingestion.")
        st.rerun()

if st.session_state.selected_conversation_id is None:
    st.info("Select a conversation or start a new one from the sidebar.")
else:
    st.header(f"Conversation ID: {st.session_state.selected_conversation_id}")

    for msg_index, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg.get("role", "assistant")):
            st.markdown(msg.get("content", ""))
            render_sources(f"history_{msg_index}", msg.get("sources", []))

    if prompt := st.chat_input("Ask me anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        job_info = start_query_job(st.session_state.selected_conversation_id, prompt)
        if not job_info or "job_id" not in job_info:
            with st.chat_message("assistant"):
                st.error("Failed to start the query workflow.")
        else:
            job_id = job_info["job_id"]
            with st.chat_message("assistant"):
                status_placeholder = st.empty()
                answer_placeholder = st.empty()
                sources_container = st.container()
                status_placeholder.info("Starting pipeline...")

                start_time = time.time()
                last_phase = None
                while True:
                    job_state = get_query_job(job_id)
                    if not job_state:
                        status_placeholder.error("Lost connection to query status.")
                        break

                    status = job_state.get("status")
                    phase = job_state.get("phase")
                    if phase and phase != last_phase:
                        label = phase if status != "running" else f"{phase}..."
                        status_placeholder.info(label)
                        last_phase = phase

                    if status == "completed":
                        result = job_state.get("result", {})
                        response_content = result.get("response", "Sorry, I encountered an error.")
                        sources = result.get("sources", [])
                        status_placeholder.empty()
                        answer_placeholder.markdown(response_content)
                        with sources_container:
                            render_sources(f"live_{job_id}", sources)
                        assistant_entry: Dict[str, Any] = {
                            "role": "assistant",
                            "content": response_content,
                            "metadata": result,
                            "sources": sources,
                        }
                        st.session_state.messages.append(assistant_entry)
                        break

                    if status == "failed":
                        error_message = job_state.get("error", "Query workflow failed.")
                        status_placeholder.error(error_message)
                        break

                    if time.time() - start_time > QUERY_POLL_TIMEOUT:
                        status_placeholder.error("Query timed out. Please try again.")
                        break

                    time.sleep(QUERY_POLL_INTERVAL_SECONDS)

viewer_source = st.session_state.get("selected_source")
if viewer_source:
    render_pdf_preview(viewer_source)

if st.session_state.ingestion_running:
    time.sleep(POLL_INTERVAL_SECONDS)
    st.rerun()





