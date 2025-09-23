# database.py
"""
Manages all interactions with the PostgreSQL database for storing and
retrieving conversation history.
"""

import asyncpg 
import json
from typing import List, Dict, Optional

class DatabaseManager():
    """Handles all database related operations for the chatbot."""

    def __init__(self, dsn: str):
        """
        Initializes the DatabaseManager.
        
        Args: 
            dsn (str): The database connection string.
        """
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
        """
        Initializes the database by creating necessary tables if they don't exist.
        """
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
        """
        Creates a new conversation in the database.

        Args:
            title (str): The title of the conversation.

        Returns:
            int: The ID of the newly created conversation.
        """
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                "INSERT INTO conversations (title) VALUES ($1) RETURNING id", title
            )
            return row['id']
    
    async def add_message(self, conversation_id: int, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Adds a new message to a specific conversation.

        Args:
            conversation_id (int): The ID of the conversation.
            role (str): The role of the message sender ('user' or 'assistant').
            content (str): The content of the message.
            metadata (dict, optional): A dictionary of metadata to store.
        """
        async with self.pool.acquire() as connection:
            metadata_json = json.dumps(metadata) if metadata else None
            await connection.execute(
                "INSERT INTO messages (conversation_id, role, content, metadata) VALUES ($1, $2, $3, $4)",
                conversation_id, role, content, metadata_json
            )

    async def get_conversations(self) -> List[Dict]:
        """Retrieves all conversations from the database."""
        async with self.pool.acquire() as connection:
            rows = await connection.fetch("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC")
            return [dict(row) for row in rows]
    
    async def get_messages(self, conversation_id: int) -> List[Dict]:
        """
        Retrieves all messages for a specific conversation.

        Args:
            conversation_id (int): The ID of the conversation to fetch messages for.

        Returns:
            List[Dict]: A list of messages, ordered by creation time.
        """
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT role, content FROM messages WHERE conversation_id = $1 ORDER BY created_at ASC",
                conversation_id
            )
            return [dict(row) for row in rows]