# ui.py
"""
This script creates a user-friendly web interface for the RAG chatbot
using the Streamlit library.
Updated to be compatible with Docker container networking.
"""
import os
import streamlit as st
import requests
import time

# --- Configuration ---
# Read the API URL from an environment variable for Docker, default for local dev
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

# --- Helper Functions ---
def get_conversations():
    """Fetches all past conversations from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/conversations")
        response.raise_for_status()
        return response.json().get("conversations", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching conversations: {e}")
        return []

def create_conversation(title):
    """Creates a new conversation through the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/conversations",
            json={"title": title}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error creating conversation: {e}")
        return None

def get_messages(conversation_id):
    """Fetches messages for a specific conversation."""
    try:
        response = requests.get(f"{API_BASE_URL}/conversations/{conversation_id}")
        response.raise_for_status()
        return response.json().get("messages", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching messages: {e}")
        return []

def post_query(conversation_id, query):
    """Posts a new query to a conversation and gets the agent's response."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/conversations/{conversation_id}/query",
            json={"query": query}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting response: {e}")
        return None

def get_documents():
    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return []
    
def delete_document(filename):
    try:
        response = requests.delete(f"{API_BASE_URL}/documents/{filename}")
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error deleting file: {e}")
        return False

def trigger_ingestion():
    try: 
        response = requests.post(f"{API_BASE_URL}/ingest")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error starting ingestion: {e}")
        return None

# --- Streamlit App Layout ---

st.set_page_config(page_title="Multi-Agent RAG Chatbot", layout="wide")

st.title("🤖 Multi-Agent RAG Chatbot")

# --- Session State Initialization ---
if "selected_conversation_id" not in st.session_state:
    st.session_state.selected_conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar for Conversation Management ---
with st.sidebar:
    st.header("Conversations")

    if st.button("➕ New Chat"):
        new_conv_title = f"New Chat {int(time.time())}"
        new_conv = create_conversation(new_conv_title)
        if new_conv:
            st.session_state.selected_conversation_id = new_conv["conversation_id"]
            st.session_state.messages = []
            st.rerun()

    st.subheader("Past Chats")
    conversations = get_conversations()
    if not conversations:
        st.write("No past conversations found.")
    else:
        for conv in conversations:
            if st.button(f"{conv['title']} (ID: {conv['id']})", key=f"conv_{conv['id']}"):
                st.session_state.selected_conversation_id = conv["id"]
                st.session_state.messages = get_messages(conv["id"])
                st.rerun()

with st.sidebar:
    st.header("📚 Knowledge Base")

    # --- Initialize session state for tracking uploads ---
    if "last_uploaded_filename" not in st.session_state:
        st.session_state.last_uploaded_filename = None

    # --- File Uploader ---
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    # Check if a file is present AND its name is different from the last one uploaded
    if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_filename:
        with st.spinner("Uploading..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{API_BASE_URL}/documents/upload", files=files)

            if response.status_code == 200:
                # IMPORTANT: Update the session state with the new filename
                st.session_state.last_uploaded_filename = uploaded_file.name
                st.success(f"File '{uploaded_file.name}' uploaded successfully!")
                st.rerun()
            else:
                # Clear the tracker on failure to allow for a retry
                st.session_state.last_uploaded_filename = None
                st.error("Upload failed.")


    # Display and Delete Files
    st.subheader("Managed Files")
    doc_list = get_documents()
    if not doc_list:
        st.write("No documents found.")
    else:
        for doc_name in doc_list:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(doc_name)
            with col2:
                if st.button("🗑️", key=f"del_{doc_name}"):
                    if delete_document(doc_name):
                        st.success(f"Deleted '{doc_name}'.")
                        st.rerun()
    
    st.divider()
    if st.button("🔄 Update Knowledge Base"):
        with st.spinner("Ingestion process started in the background."):
            result = trigger_ingestion()
            if result:
                st.success(result.get("message"))

# --- Main Chat Interface ---
if st.session_state.selected_conversation_id is None:
    st.info("Select a conversation or start a new one from the sidebar.")
else:
    st.header(f"Conversation ID: {st.session_state.selected_conversation_id}")

    # Display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle new user input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message to session state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_data = post_query(st.session_state.selected_conversation_id, prompt)

            if response_data:
                response_content = response_data.get("response", "Sorry, I encountered an error.")
                st.markdown(response_content)
                # Add assistant response to session state
                st.session_state.messages.append({"role": "assistant", "content": response_content})
            else:
                st.error("Failed to get a response from the backend.")

