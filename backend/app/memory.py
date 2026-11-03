"""User financial profiles and user-controlled memory notes.

Users can always see, edit and delete every note the co-pilot may use; the
co-pilot never writes memory by itself. Profile JSON and note content are
passed through the configured cipher so they can be encrypted at rest.
"""

from __future__ import annotations

import json
import re
from datetime import date
from decimal import Decimal
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.db import Database, utc_now
from app.finance import MAX_AMOUNT
from app.security import Cipher, PlaintextCipher

USER_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{2,31}$"
_USER_ID = re.compile(USER_ID_PATTERN)
MAX_NOTES_PER_USER = 100
MAX_NOTE_LENGTH = 500

Money = Annotated[Decimal, Field(ge=0, le=MAX_AMOUNT)]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
NoteText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_NOTE_LENGTH)
]


class Debt(BaseModel):
    """A debt account and its monthly repayment."""

    model_config = ConfigDict(extra="forbid")

    name: ShortText
    balance: Money
    apr: Decimal = Field(ge=0, le=1, description="Annual rate as a fraction: 0.249 = 24.9%.")
    monthly_payment: Money


class Goal(BaseModel):
    """A dated savings or debt-repayment goal."""

    model_config = ConfigDict(extra="forbid")

    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
    type: Literal["debt_free", "safety_net"]
    target_amount: Money
    target_date: date = Field(ge=date(2000, 1, 1), le=date(2100, 12, 31))


class UserProfile(BaseModel):
    """Financial details and goals a user has chosen to save."""

    model_config = ConfigDict(extra="forbid")

    name: ShortText
    monthly_take_home_income: Money
    monthly_essentials: Money
    savings: Money
    debts: list[Debt] = Field(default_factory=list, max_length=20)
    goals: list[Goal] = Field(default_factory=list, max_length=20)


class MemoryNote(BaseModel):
    """A user-visible note remembered by the co-pilot."""

    note_id: str
    content: str


class NoteCreate(BaseModel):
    """Content for a new remembered note."""

    model_config = ConfigDict(extra="forbid")

    content: NoteText


class NoteUpdate(NoteCreate):
    """Replacement content for a remembered note."""


class MemoryResponse(BaseModel):
    """All remembered notes belonging to a user."""

    notes: list[MemoryNote]


class NoteDeletedResponse(BaseModel):
    """Confirmation that a remembered note was deleted."""

    deleted: bool


class ProfileNotFoundError(LookupError):
    """Raised when a requested profile or note does not exist."""


class NoteLimitError(ValueError):
    """Raised when a user already has the maximum number of notes."""


def validate_user_id(user_id: str) -> str:
    """Ensure a user ID is a safe, canonical identifier."""
    if not _USER_ID.fullmatch(user_id):
        raise ValueError(
            "user_id must be 3-32 lower-case letters, numbers, '_' or '-', "
            "starting with a letter or number."
        )
    return user_id


class ProfileStore:
    """Persist profiles and memory notes in SQLite."""

    def __init__(self, database: Database, cipher: Cipher | None = None) -> None:
        self._db = database
        self._cipher = cipher or PlaintextCipher()

    def get_profile(self, user_id: str) -> UserProfile:
        with self._db.connect() as connection:
            row = connection.execute(
                "SELECT data FROM profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            raise ProfileNotFoundError(f"Profile '{user_id}' was not found.")
        return UserProfile.model_validate_json(self._cipher.decrypt(row["data"]))

    def find_profile(self, user_id: str) -> UserProfile | None:
        try:
            return self.get_profile(user_id)
        except ProfileNotFoundError:
            return None

    def put_profile(self, user_id: str, profile: UserProfile) -> UserProfile:
        """Create or replace a user's profile. Memory notes are unaffected."""
        data = self._cipher.encrypt(profile.model_dump_json())
        with self._db.connect() as connection:
            connection.execute(
                "INSERT INTO profiles (user_id, data, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET data = excluded.data, "
                "updated_at = excluded.updated_at",
                (user_id, data, utc_now()),
            )
        return profile

    def get_notes(self, user_id: str) -> list[MemoryNote]:
        with self._db.connect() as connection:
            rows = connection.execute(
                "SELECT id, content FROM memory_notes WHERE user_id = ? ORDER BY created_at, id",
                (user_id,),
            ).fetchall()
        return [
            MemoryNote(note_id=row["id"], content=self._cipher.decrypt(row["content"]))
            for row in rows
        ]

    def add_note(self, user_id: str, content: str) -> MemoryNote:
        note = MemoryNote(note_id=str(uuid4()), content=content)
        now = utc_now()
        with self._db.connect() as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM memory_notes WHERE user_id = ?", (user_id,)
            ).fetchone()[0]
            if count >= MAX_NOTES_PER_USER:
                raise NoteLimitError(
                    f"You can save up to {MAX_NOTES_PER_USER} notes. Delete one to add another."
                )
            connection.execute(
                "INSERT INTO memory_notes (id, user_id, content, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (note.note_id, user_id, self._cipher.encrypt(content), now, now),
            )
        return note

    def update_note(self, user_id: str, note_id: str, content: str) -> MemoryNote:
        with self._db.connect() as connection:
            cursor = connection.execute(
                "UPDATE memory_notes SET content = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (self._cipher.encrypt(content), utc_now(), note_id, user_id),
            )
        if cursor.rowcount == 0:
            raise ProfileNotFoundError(f"Memory note '{note_id}' was not found.")
        return MemoryNote(note_id=note_id, content=content)

    def delete_note(self, user_id: str, note_id: str) -> None:
        with self._db.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM memory_notes WHERE id = ? AND user_id = ?", (note_id, user_id)
            )
        if cursor.rowcount == 0:
            raise ProfileNotFoundError(f"Memory note '{note_id}' was not found.")

    def delete_all_notes(self, user_id: str) -> int:
        with self._db.connect() as connection:
            cursor = connection.execute("DELETE FROM memory_notes WHERE user_id = ?", (user_id,))
        return cursor.rowcount

    def import_record(self, user_id: str, profile: UserProfile, notes: list[MemoryNote]) -> None:
        """Insert a legacy profile and its notes, keeping existing data intact."""
        now = utc_now()
        with self._db.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO profiles (user_id, data, updated_at) VALUES (?, ?, ?)",
                (user_id, self._cipher.encrypt(profile.model_dump_json()), now),
            )
            for note in notes:
                connection.execute(
                    "INSERT OR IGNORE INTO memory_notes "
                    "(id, user_id, content, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (note.note_id, user_id, self._cipher.encrypt(note.content), now, now),
                )


def demo_profile() -> tuple[UserProfile, list[str]]:
    """The Alex demo profile used for local development only."""
    profile = UserProfile.model_validate(
        json.loads(
            """{
            "name": "Alex",
            "monthly_take_home_income": "3600",
            "monthly_essentials": "2000",
            "savings": "3200",
            "debts": [{"name": "Credit card", "balance": "2450", "apr": "0.249",
                       "monthly_payment": "350"}],
            "goals": [
                {"title": "A cushion for the unexpected", "type": "safety_net",
                 "target_amount": "6000", "target_date": "2027-06-01"},
                {"title": "A fresh, debt-free start", "type": "debt_free",
                 "target_amount": "2450", "target_date": "2027-06-01"}
            ]
        }"""
        )
    )
    return profile, ["Wants progress without putting life on hold"]
