"""User accounts and bearer-token sessions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.db import Database, utc_now
from app.security import hash_password, hash_token, new_session_token, verify_password


class UserExistsError(ValueError):
    """Raised when registering a user ID that is already taken."""


class UserNotFoundError(LookupError):
    """Raised when an operation targets a user that does not exist."""


class UserStore:
    """Create users, verify passwords and manage hashed session tokens."""

    def __init__(self, database: Database, session_ttl: timedelta) -> None:
        self._db = database
        self._ttl = session_ttl

    def exists(self, user_id: str) -> bool:
        with self._db.connect() as connection:
            row = connection.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone()
        return row is not None

    def create_user(self, user_id: str, password: str | None) -> None:
        """Create a user. A `None` password creates an account that cannot log in."""
        password_hash = hash_password(password) if password is not None else None
        with self._db.connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO users (id, password_hash, created_at) VALUES (?, ?, ?)",
                (user_id, password_hash, utc_now()),
            )
        if cursor.rowcount == 0:
            raise UserExistsError("That user ID is not available.")

    def set_password(self, user_id: str, password: str) -> None:
        """Replace a password and revoke every existing session."""
        with self._db.connect() as connection:
            cursor = connection.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (hash_password(password), user_id),
            )
            if cursor.rowcount == 0:
                raise UserNotFoundError(f"User '{user_id}' was not found.")
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    def authenticate(self, user_id: str, password: str) -> bool:
        """Verify credentials in roughly constant time whether or not the user exists."""
        with self._db.connect() as connection:
            row = connection.execute(
                "SELECT password_hash FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return verify_password(password, row["password_hash"] if row else None)

    def create_session(self, user_id: str) -> tuple[str, datetime]:
        """Issue a new bearer token. Only its hash is stored."""
        token = new_session_token()
        now = datetime.now(UTC)
        expires_at = now + self._ttl
        with self._db.connect() as connection:
            connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now.isoformat(),))
            connection.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (hash_token(token), user_id, now.isoformat(), expires_at.isoformat()),
            )
        return token, expires_at

    def resolve_session(self, token: str) -> str | None:
        """Return the user ID for a valid, unexpired token."""
        token_hash = hash_token(token)
        now = datetime.now(UTC).isoformat()
        with self._db.connect() as connection:
            row = connection.execute(
                "SELECT user_id, expires_at FROM sessions WHERE token_hash = ?", (token_hash,)
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] <= now:
                connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
                return None
        return str(row["user_id"])

    def revoke_session(self, token: str) -> None:
        with self._db.connect() as connection:
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (hash_token(token),))

    def delete_user(self, user_id: str) -> None:
        """Delete a user and, through cascades, all of their data."""
        with self._db.connect() as connection:
            cursor = connection.execute("DELETE FROM users WHERE id = ?", (user_id,))
        if cursor.rowcount == 0:
            raise UserNotFoundError(f"User '{user_id}' was not found.")
