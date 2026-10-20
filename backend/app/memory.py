"""User profile and editable memory-note storage for the NetValor API."""

import json
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Debt(BaseModel):
    """A debt account and its monthly repayment."""

    name: str = Field(min_length=1)
    balance: Decimal = Field(ge=0)
    apr: Decimal = Field(ge=0, le=1)
    monthly_payment: Decimal = Field(ge=0)


class Goal(BaseModel):
    """A dated savings or debt-repayment goal."""

    title: str = Field(min_length=1)
    type: Literal["debt_free", "safety_net"]
    target_amount: Decimal = Field(ge=0)
    target_date: date


class UserProfile(BaseModel):
    """Financial details and goals a user has chosen to save."""

    name: str = Field(min_length=1)
    monthly_take_home_income: Decimal = Field(ge=0)
    monthly_essentials: Decimal = Field(ge=0)
    savings: Decimal = Field(ge=0)
    debts: list[Debt] = Field(default_factory=list)
    goals: list[Goal] = Field(default_factory=list)


class MemoryNote(BaseModel):
    """A user-visible note remembered by the co-pilot."""

    note_id: str
    content: str = Field(min_length=1)


class NoteCreate(BaseModel):
    """Content for a new remembered note."""

    content: str = Field(min_length=1)


class NoteUpdate(BaseModel):
    """Replacement content for a remembered note."""

    content: str = Field(min_length=1)


class MemoryResponse(BaseModel):
    """All remembered notes belonging to a user."""

    notes: list[MemoryNote]


class NoteDeletedResponse(BaseModel):
    """Confirmation that a remembered note was deleted."""

    deleted: bool


@dataclass
class _ProfileRecord:
    profile: UserProfile
    notes: list[MemoryNote]


class ProfileNotFoundError(LookupError):
    """Raised when a requested profile or note does not exist."""


class JsonProfileStore:
    """Persist profiles and user-controlled memory in one JSON file per user."""

    def __init__(self, profiles_dir: Path) -> None:
        self._profiles_dir = Path(profiles_dir)

    def seed_demo_user(self) -> None:
        """Create the Alex demo profile and note unless the user already exists."""
        if self._profile_path("alex").exists():
            return
        profile = UserProfile(
            name="Alex",
            monthly_take_home_income=Decimal("3600"),
            monthly_essentials=Decimal("2000"),
            savings=Decimal("3200"),
            debts=[
                Debt(
                    name="Credit card",
                    balance=Decimal("2450"),
                    apr=Decimal("0.249"),
                    monthly_payment=Decimal("350"),
                )
            ],
            goals=[
                Goal(
                    title="A cushion for the unexpected",
                    type="safety_net",
                    target_amount=Decimal("6000"),
                    target_date=date(2027, 6, 1),
                ),
                Goal(
                    title="A fresh, debt-free start",
                    type="debt_free",
                    target_amount=Decimal("2450"),
                    target_date=date(2027, 6, 1),
                ),
            ],
        )
        self._write_record(
            "alex",
            _ProfileRecord(
                profile=profile,
                notes=[
                    MemoryNote(
                        note_id=str(uuid4()),
                        content="Wants progress without putting life on hold",
                    )
                ],
            ),
        )

    def get_profile(self, user_id: str) -> UserProfile:
        """Load a user's profile."""
        return self._read_record(user_id).profile

    def update_profile(self, user_id: str, profile: UserProfile) -> UserProfile:
        """Create or replace a profile while preserving existing remembered notes."""
        try:
            notes = self._read_record(user_id).notes
        except ProfileNotFoundError:
            notes = []
        self._write_record(user_id, _ProfileRecord(profile, notes))
        return profile

    def get_notes(self, user_id: str) -> list[MemoryNote]:
        """Return a copy of a user's remembered notes."""
        return list(self._read_record(user_id).notes)

    def add_note(self, user_id: str, content: str) -> MemoryNote:
        """Create and save a remembered note."""
        record = self._read_record(user_id)
        note = MemoryNote(note_id=str(uuid4()), content=content)
        record.notes.append(note)
        self._write_record(user_id, record)
        return note

    def update_note(self, user_id: str, note_id: str, content: str) -> MemoryNote:
        """Edit a remembered note by its ID."""
        record = self._read_record(user_id)
        for index, note in enumerate(record.notes):
            if note.note_id == note_id:
                updated = MemoryNote(note_id=note_id, content=content)
                record.notes[index] = updated
                self._write_record(user_id, record)
                return updated
        raise ProfileNotFoundError(f"Memory note '{note_id}' was not found.")

    def delete_note(self, user_id: str, note_id: str) -> None:
        """Delete a remembered note by its ID."""
        record = self._read_record(user_id)
        for index, note in enumerate(record.notes):
            if note.note_id == note_id:
                del record.notes[index]
                self._write_record(user_id, record)
                return
        raise ProfileNotFoundError(f"Memory note '{note_id}' was not found.")

    def _profile_path(self, user_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", user_id):
            raise ValueError("user_id may contain only letters, numbers, '_' and '-'.")
        return self._profiles_dir / f"{user_id}.json"

    def _read_record(self, user_id: str) -> _ProfileRecord:
        path = self._profile_path(user_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ProfileNotFoundError(f"Profile '{user_id}' was not found.") from None
        return _ProfileRecord(
            profile=UserProfile.model_validate(data["profile"]),
            notes=[MemoryNote.model_validate(note) for note in data["notes"]],
        )

    def _write_record(self, user_id: str, record: _ProfileRecord) -> None:
        path = self._profile_path(user_id)
        self._profiles_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "profile": json.loads(record.profile.model_dump_json()),
            "notes": [json.loads(note.model_dump_json()) for note in record.notes],
        }
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary_path.replace(path)