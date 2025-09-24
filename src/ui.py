# ui.py
"""
Streamlit UI for the Multi-Agent RAG Chatbot.
"""
import os
import time
from typing import Any, Dict, List

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
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


def get_messages(conversation_id: int) -> List[Dict[str, Any]]:
    response = _safe_request("GET", f"/conversations/{conversation_id}")
    if not response:
        return []
    return response.json().get("messages", [])


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

with st.sidebar:
    st.header("Conversations")
    if st.button("New Chat"):
        new_conv_title = f"New Chat {int(time.time())}"
        new_conv = create_conversation(new_conv_title)
        if new_conv:
            st.session_state.selected_conversation_id = new_conv["conversation_id"]
            st.session_state.messages = []
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
                st.rerun()

with st.sidebar:
    st.header("Knowledge Base")

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

    for msg in st.session_state.messages:
        with st.chat_message(msg.get("role", "assistant")):
            st.markdown(msg.get("content", ""))

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
                        status_placeholder.empty()
                        answer_placeholder.markdown(response_content)
                        assistant_entry: Dict[str, Any] = {"role": "assistant", "content": response_content}
                        metadata = result.get("metadata")
                        if metadata is not None:
                            assistant_entry["metadata"] = metadata
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

if st.session_state.ingestion_running:
    time.sleep(POLL_INTERVAL_SECONDS)
    st.rerun()
