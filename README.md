# Advanced Multi-Agent RAG Workflow

This project implements a high-performance, modular Retrieval-Augmented Generation (RAG) system designed for accuracy, efficiency, and scalability. It is architected as a sophisticated, multi-agent backend service that leverages an asynchronous, event-driven workflow to deliver high-quality, fact-checked, and context-aware answers.

The system is designed to be a stateful, conversational agent, capable of remembering chat history to answer follow-up questions effectively.

## Core Philosophy: Evidence-Based Design

Every architectural decision in this project is deliberately informed by empirical data and established best practices from leading research:

-   **Component Design:** System modules are direct implementations of the "Best Performance Practice" configurations identified in the research paper *"Searching for Best Practices in Retrieval-Augmented Generation"*. This includes specific choices for embedding (`BAAI/llm-embedder`), retrieval (`Hybrid Search with HyDE`), and reranking (`cross-encoder`).
-   **Structural Paradigm:** The system is modeled as a multi-agent collaboration, drawing inspiration from the specialized agent roles outlined in *"DocAgent: A Multi-Agent System for Automated Code Documentation"*.
-   **Implementation Pattern:** The underlying orchestration leverages the powerful, asynchronous, event-driven workflow patterns demonstrated in LlamaIndex deep research code examples.

## System Architecture

The project is built on three foundational principles: a custom event-driven engine, a clear separation of workflows, and specialized agent roles.

### 1. Workflows

The system operates in two distinct modes: an offline ingestion process and a real-time query process.

#### Ingestion Workflow (Offline)

This workflow is responsible for processing source documents and building a persistent, queryable knowledge base. It loads documents, splits them into chunks, generates embeddings, and stores them in a ChromaDB vector store. It also creates a BM25 index for keyword search.

#### Query Workflow (Real-Time)

This is the interactive workflow that processes a user's query to generate a high-quality, evidence-based, and self-corrected answer. It is designed to be stateful, maintaining conversation history to understand context.

### 2. Multi-Agent Specialization

The Query Workflow is a collaboration of specialized agents, each with a single responsibility. This "separation of concerns" enhances modularity, testability, and clarity.

-   `QueryPlannerAgent`: The "Gatekeeper." Analyzes conversation history and the latest query to create a standalone, context-aware question. It also determines if retrieval is necessary and generates a hypothetical document (HyDE) to improve search relevance.
-   `SearcherAgent`: The "Librarian." Executes a hybrid search using both dense (vector) and sparse (BM25) retrieval methods to find a diverse set of relevant documents.
-   `RerankerAgent`: The "Expert Analyst." Uses a powerful cross-encoder model to re-score and re-order the retrieved documents for maximum relevance, passing only the best candidates forward.
-   `WriterAgent`: The "Author." Synthesizes the top-ranked context into a coherent, final answer. It is capable of receiving feedback to perform revisions.
-   `VerifierAgent`: The "Fact-Checker." Checks the generated answer against the source documents to ensure faithfulness and provides specific feedback if inaccuracies are found, triggering a rewrite loop.
-   `DirectAnswerAgent`: The "Conversationalist." Handles non-retrieval queries by providing a concise, natural answer.

## Setup and Installation

Follow these steps to set up and run the project locally.

### Prerequisites

-   Python 3.10+
-   Access to a GPU is highly recommended for the embedding and LLM models.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```
### 2. Set up your virtual environment and install the dependencies 

```bash
pip install -r requirements.txt
```

### 3. Set up local LLM with Ollama
Download and install for your OS. Pull the models specified in config.py
```bash
ollama pull qwen3:8b
ollama pull qwen3:1.7b
```

### 4. Add your source documents into the ./data directory before running the chatbot. 

## Usage

### 1. Run Data Ingestion 
Before you can ask questions, it is recommended to process your source documents. Run the ingestion workflow from the command line as such

```bash
python chatbot.py
/ingest
```

You can now ask questions after the ingestion is done processing.

### Special commands 
- /ingest: Type `/ingest` during a chat session at any time to re-run the data ingestion process and renew the knowledge base.
- exit: Type `exit` to end the chat session.

### 2. Preliminary Performance Tests
In order to make sure that your system is to up to par to run the pipeline efficiently, a test.py script has been left with the timings associated in a separate text file. 
Please run them and make sure that your system isn't taking too long to process the list of queries. 

## Extra

### Configuration 
A configuration file with the key params has been left so that they can be quickly accessed and adjusted as needed:
- Paths: `DATA_DIR`, `CHROMA_PERSIST_DIR`, `NODES_PATH`
- Chunking: `CHUNK_SIZE`, `CHUNK_OVERLAP`
- Models: `EMBEDDING_MODEL_NAME`, `LLM_MODEL`, `UTILITY_MODEL`
- Agent Behavior: `USE_VERIFIER`, `MAX_REWRITES`

### Future Work
- `Persistent Chat History`: Implement a database (SQL) to store and retrieve chat sessions, allowing users to pick up and continue conversations across multiple application runs.
- `Web Interface`: Build a minimalist, user-friendly frontend using a framework like Streamlit or FastAPI + React in order to allow easy interaction with the agent.
- `Containerization`: Package the entire application using Docker for easy and scalable deployment.
