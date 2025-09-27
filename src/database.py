# database.py
"""
Manages all interactions with the PostgreSQL database for storing and
retrieving conversation history.
"""

import asyncpg
import json
from typing import Dict, List, Optional


class DatabaseManager():
    """Handles all database related operations for the chatbot."""

    def __init__(self, dsn: str):
        """Initializes the DatabaseManager."""
        self.dsn = dsn
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Creates a connection pool to the database."""
        if not self.pool:
            try:
                self.pool = await asyncpg.create_pool(dsn=self.dsn)
                print("--- Database connection pool created successfully ---")
            except Exception as e:
                print(f"--- Failed to connect to the database: {e} ---")
                raise

    async def close(self):
        """Closes the connection pool."""
        if self.pool:
            await self.pool.close()
            print("--- Database connection pool closed. ---")

    async def init_db(self):
        """Initializes the database tables if they don't exist."""
        async with self.pool.acquire() as connection:
            await connection.execute("""
                CREATE TABLE IF NOT EXISTS conversations(
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
            """)
            await connection.execute("""
                CREATE TABLE IF NOT EXISTS messages(
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
            """)

        print("--- Database tables initialized ---")

    async def create_conversation(self, title: str) -> int:
        """Creates a new conversation and returns its ID."""
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                "INSERT INTO conversations (title) VALUES ($1) RETURNING id", title
            )
            return row['id']

    async def add_message(self, conversation_id: int, role: str, content: str, metadata: Optional[Dict] = None):
        """Adds a message (with optional metadata) to a conversation."""
        async with self.pool.acquire() as connection:
            metadata_json = json.dumps(metadata) if metadata else None
            await connection.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES ($1, $2, $3, $4)",
                conversation_id, role, content, metadata_json
            )

    async def get_conversations(self) -> List[Dict]:
        """Retrieves all conversations."""
        async with self.pool.acquire() as connection:
            rows = await connection.fetch("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC")
            return [dict(row) for row in rows]

    async def get_messages(self, conversation_id: int) -> List[Dict]:
        """Retrieves all messages (including metadata) for the given conversation."""
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT role, content, metadata FROM messages WHERE conversation_id = $1 ORDER BY created_at ASC",
                conversation_id
            )

        messages: List[Dict] = []
        for row in rows:
            metadata = row.get('metadata')
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = None
            messages.append({
                'role': row.get('role'),
                'content': row.get('content'),
                'metadata': metadata or {}
            })
        return messages

