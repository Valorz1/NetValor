"""Conversation persistence.

Each turn (user message + assistant reply) is appended in one transaction,
so concurrent requests on the same conversation can never overwrite each
other's messages, and a crash can never leave half a turn behind.
"""

from __future__ import annotations

import re
from typing import Any, Literal
from uuid import uuid4

from app.db import Database, utc_now
from app.security import Cipher, PlaintextCipher

MessageStatus = Literal["complete", "incomplete", "clarification"]
MAX_MESSAGES_PER_CONVERSATION = 200
_CONVERSATION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ConversationNotFoundError(LookupError):
    """Raised when a conversation does not exist for this user."""


class ConversationFullError(ValueError):
    """Raised when a conversation has reached its message limit."""


def validate_conversation_id(conversation_id: str) -> str:
    if not _CONVERSATION_ID.fullmatch(conversation_id):
        raise ValueError("conversation_id may contain only letters, numbers, '_' and '-'.")
    return conversation_id


class ConversationStore:
    """Store conversations and their messages in SQLite."""

    def __init__(self, database: Database, cipher: Cipher | None = None) -> None:
        self._db = database
        self._cipher = cipher or PlaintextCipher()

    @staticmethod
    def create_id() -> str:
        return str(uuid4())

    def exists(self, user_id: str, conversation_id: str) -> bool:
        with self._db.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
        return row is not None

    def get(self, user_id: str, conversation_id: str) -> dict[str, Any]:
        """Return a conversation with all messages, scoped to its owner."""
        with self._db.connect() as connection:
            row = connection.execute(
                "SELECT id, user_id, created_at, updated_at FROM conversations "
                "WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
            if row is None:
                raise ConversationNotFoundError(f"Conversation '{conversation_id}' was not found.")
            messages = connection.execute(
                "SELECT role, content, status, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        return {
            "conversation_id": row["id"],
            "user_id": row["user_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "messages": [
                {
                    "role": message["role"],
                    "content": self._cipher.decrypt(message["content"]),
                    "status": message["status"],
                    "created_at": message["created_at"],
                }
                for message in messages
            ],
        }

    def list(self, user_id: str) -> list[dict[str, Any]]:
        """Summaries of a user's conversations, most recently updated first."""
        with self._db.connect() as connection:
            rows = connection.execute(
                "SELECT c.id, c.created_at, c.updated_at, "
                "  (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS n, "
                "  (SELECT content FROM messages m WHERE m.conversation_id = c.id "
                "   ORDER BY m.id DESC LIMIT 1) AS last "
                "FROM conversations c WHERE c.user_id = ? ORDER BY c.updated_at DESC",
                (user_id,),
            ).fetchall()
        return [
            {
                "conversation_id": row["id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "message_count": row["n"],
                "last_message": self._cipher.decrypt(row["last"]) if row["last"] else "",
            }
            for row in rows
        ]

    def append_turn(
        self,
        user_id: str,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        status: MessageStatus = "complete",
    ) -> None:
        """Atomically add one exchange, creating the conversation if needed."""
        now = utc_now()
        with self._db.connect() as connection:
            owner = connection.execute(
                "SELECT user_id FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if owner is None:
                connection.execute(
                    "INSERT INTO conversations (id, user_id, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (conversation_id, user_id, now, now),
                )
            elif owner["user_id"] != user_id:
                raise ConversationNotFoundError(f"Conversation '{conversation_id}' was not found.")
            count = connection.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()[0]
            if count + 2 > MAX_MESSAGES_PER_CONVERSATION:
                raise ConversationFullError(
                    "This conversation is full. Please start a new conversation."
                )
            connection.executemany(
                "INSERT INTO messages (conversation_id, role, content, status, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (conversation_id, "user", self._cipher.encrypt(user_message), "complete", now),
                    (
                        conversation_id,
                        "assistant",
                        self._cipher.encrypt(assistant_message),
                        status,
                        now,
                    ),
                ],
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
            )

    def message_count(self, user_id: str, conversation_id: str) -> int:
        with self._db.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM messages m JOIN conversations c ON c.id = m.conversation_id "
                "WHERE c.id = ? AND c.user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
        return int(row[0])

    def delete(self, user_id: str, conversation_id: str) -> None:
        with self._db.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
        if cursor.rowcount == 0:
            raise ConversationNotFoundError(f"Conversation '{conversation_id}' was not found.")

    def import_conversation(self, user_id: str, conversation: dict[str, Any]) -> bool:
        """Insert a legacy JSON conversation unless it already exists."""
        conversation_id = validate_conversation_id(str(conversation["conversation_id"]))
        with self._db.connect() as connection:
            if connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone():
                return False
            created = str(conversation.get("created_at") or utc_now())
            updated = str(conversation.get("updated_at") or created)
            connection.execute(
                "INSERT INTO conversations (id, user_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (conversation_id, user_id, created, updated),
            )
            connection.executemany(
                "INSERT INTO messages (conversation_id, role, content, status, created_at) "
                "VALUES (?, ?, ?, 'complete', ?)",
                [
                    (
                        conversation_id,
                        message["role"],
                        self._cipher.encrypt(message["content"]),
                        updated,
                    )
                    for message in conversation.get("messages", [])
                    if message.get("role") in {"user", "assistant"}
                    and isinstance(message.get("content"), str)
                ],
            )
        return True
