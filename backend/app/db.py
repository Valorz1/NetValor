"""SQLite connection management and schema migrations.

SQLite (standard library) gives NetValor atomic transactions, row-level
appends for concurrent chats, foreign-key cascades for account deletion and
crash-safe writes, without running a separate database server.

Migrations are numbered SQL files in `app/migrations/`. Each is applied once,
inside a transaction, and recorded in `PRAGMA user_version`.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
_MIGRATION_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


def utc_now() -> str:
    """ISO-8601 UTC timestamp used for all stored dates."""
    return datetime.now(UTC).isoformat(timespec="microseconds")


class Database:
    """Open short-lived connections to one SQLite database file.

    A fresh connection per unit of work keeps the class thread-safe for
    FastAPI's worker threads; SQLite connections are cheap to open.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection inside a transaction that commits on success."""
        connection = sqlite3.connect(self.path, timeout=10.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 10000")
            with connection:
                yield connection
        finally:
            connection.close()

    def initialise(self) -> None:
        """Create the database file with private permissions and migrate it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        created = not self.path.exists()
        connection = sqlite3.connect(self.path, timeout=10.0)
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA foreign_keys = ON")
            self._migrate(connection)
        finally:
            connection.close()
        if created and os.name == "posix":
            os.chmod(self.path, 0o600)

    @staticmethod
    def _migrations() -> list[tuple[int, Path]]:
        found = []
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            match = _MIGRATION_NAME.match(path.name)
            if not match:
                raise RuntimeError(f"Unexpected migration file name: {path.name}")
            found.append((int(match.group(1)), path))
        versions = [version for version, _ in found]
        if versions != list(range(1, len(versions) + 1)):
            raise RuntimeError("Migrations must be numbered consecutively from 0001.")
        return found

    def _migrate(self, connection: sqlite3.Connection) -> None:
        current = connection.execute("PRAGMA user_version").fetchone()[0]
        migrations = self._migrations()
        if current > len(migrations):
            raise RuntimeError(
                f"Database schema version {current} is newer than this code "
                f"supports ({len(migrations)}). Refusing to start."
            )
        for version, path in migrations:
            if version <= current:
                continue
            logger.info("Applying database migration %s", path.name)
            script = path.read_text(encoding="utf-8")
            # executescript() commits any open transaction first, so the
            # explicit BEGIN/COMMIT makes each migration all-or-nothing.
            try:
                connection.executescript(
                    f"BEGIN;\n{script}\nPRAGMA user_version = {version};\nCOMMIT;"
                )
            except sqlite3.Error:
                connection.rollback()
                raise

    def schema_version(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    def ping(self) -> bool:
        """Return True when the database answers a trivial query."""
        try:
            with self.connect() as connection:
                connection.execute("SELECT 1").fetchone()
        except sqlite3.Error:
            return False
        return True
