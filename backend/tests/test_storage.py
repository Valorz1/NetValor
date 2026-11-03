"""Persistence: migrations, ownership, atomicity, concurrency and encryption."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import timedelta
from pathlib import Path

import pytest

from app.cli import import_json
from app.conversations import (
    MAX_MESSAGES_PER_CONVERSATION,
    ConversationFullError,
    ConversationNotFoundError,
    ConversationStore,
)
from app.db import Database
from app.memory import (
    MAX_NOTES_PER_USER,
    NoteLimitError,
    ProfileNotFoundError,
    ProfileStore,
    UserProfile,
    validate_user_id,
)
from app.security import FernetCipher, generate_encryption_key
from app.users import UserExistsError, UserStore
from tests.conftest import SAMPLE_PROFILE, make_settings


@pytest.fixture
def database(tmp_path: Path) -> Database:
    db = Database(tmp_path / "store.db")
    db.initialise()
    return db


@pytest.fixture
def users(database: Database) -> UserStore:
    store = UserStore(database, timedelta(hours=1))
    store.create_user("sam", "a long passphrase")
    store.create_user("alex", "another passphrase")
    return store


def test_migrations_apply_once_and_record_version(database: Database) -> None:
    assert database.schema_version() == 1
    database.initialise()  # idempotent
    assert database.schema_version() == 1


def test_newer_schema_than_code_refuses_to_start(tmp_path: Path) -> None:
    path = tmp_path / "future.db"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version = 99")
    connection.close()

    with pytest.raises(RuntimeError, match="newer than this code"):
        Database(path).initialise()


def test_database_uses_wal_and_foreign_keys(database: Database) -> None:
    with database.connect() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


@pytest.mark.parametrize("user_id", ["sam", "alex_2", "a-b-c"])
def test_valid_user_ids(user_id: str) -> None:
    assert validate_user_id(user_id) == user_id


@pytest.mark.parametrize(
    "user_id", ["", "ab", "Sam", "../etc", "sam/../x", "a" * 33, "_sam", "sam!"]
)
def test_invalid_user_ids(user_id: str) -> None:
    with pytest.raises(ValueError):
        validate_user_id(user_id)


# --- Users and sessions ------------------------------------------------------


def test_duplicate_user_is_rejected(users: UserStore) -> None:
    with pytest.raises(UserExistsError):
        users.create_user("sam", "whatever passphrase")


def test_session_lifecycle(users: UserStore) -> None:
    token, _ = users.create_session("sam")

    assert users.resolve_session(token) == "sam"
    users.revoke_session(token)
    assert users.resolve_session(token) is None


def test_tokens_are_not_stored_in_plaintext(users: UserStore, database: Database) -> None:
    token, _ = users.create_session("sam")

    raw = b"".join(
        path.read_bytes() for path in database.path.parent.glob(database.path.name + "*")
    )
    assert token.encode() not in raw


def test_expired_session_is_rejected_and_removed(database: Database) -> None:
    store = UserStore(database, timedelta(seconds=-1))
    store.create_user("sam", "a long passphrase")
    token, _ = store.create_session("sam")

    assert store.resolve_session(token) is None
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_password_change_revokes_sessions(users: UserStore) -> None:
    token, _ = users.create_session("sam")
    users.set_password("sam", "a brand new passphrase")

    assert users.resolve_session(token) is None
    assert users.authenticate("sam", "a brand new passphrase")
    assert not users.authenticate("sam", "a long passphrase")


def test_account_without_password_cannot_log_in(database: Database) -> None:
    store = UserStore(database, timedelta(hours=1))
    store.create_user("imported", None)

    assert not store.authenticate("imported", "")
    assert not store.authenticate("imported", "anything")


def test_deleting_a_user_cascades_to_all_their_data(users: UserStore, database: Database) -> None:
    profiles = ProfileStore(database)
    conversations = ConversationStore(database)
    profiles.put_profile("sam", UserProfile.model_validate(SAMPLE_PROFILE))
    profiles.add_note("sam", "note")
    conversations.append_turn("sam", "c1", "hi", "hello")
    users.create_session("sam")

    users.delete_user("sam")

    with database.connect() as connection:
        for table in ("profiles", "memory_notes", "conversations", "messages", "sessions"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert users.exists("alex")


# --- Profiles and notes --------------------------------------------------------


def test_profile_and_notes_are_scoped_to_their_owner(users: UserStore, database: Database) -> None:
    profiles = ProfileStore(database)
    profiles.put_profile("sam", UserProfile.model_validate(SAMPLE_PROFILE))
    note = profiles.add_note("sam", "Paid monthly on the 28th")

    with pytest.raises(ProfileNotFoundError):
        profiles.get_profile("alex")
    assert profiles.get_notes("alex") == []
    with pytest.raises(ProfileNotFoundError):
        profiles.update_note("alex", note.note_id, "hijacked")
    with pytest.raises(ProfileNotFoundError):
        profiles.delete_note("alex", note.note_id)
    assert profiles.get_notes("sam")[0].content == "Paid monthly on the 28th"


def test_replacing_a_profile_keeps_notes(users: UserStore, database: Database) -> None:
    profiles = ProfileStore(database)
    profiles.add_note("sam", "keep me")
    profiles.put_profile("sam", UserProfile.model_validate(SAMPLE_PROFILE))
    profiles.put_profile("sam", UserProfile.model_validate({**SAMPLE_PROFILE, "name": "Samira"}))

    assert profiles.get_profile("sam").name == "Samira"
    assert [note.content for note in profiles.get_notes("sam")] == ["keep me"]


def test_note_limit(users: UserStore, database: Database) -> None:
    profiles = ProfileStore(database)
    for index in range(MAX_NOTES_PER_USER):
        profiles.add_note("sam", f"note {index}")

    with pytest.raises(NoteLimitError):
        profiles.add_note("sam", "one too many")


def test_personal_data_is_encrypted_at_rest(users: UserStore, database: Database) -> None:
    cipher = FernetCipher([generate_encryption_key()])
    profiles = ProfileStore(database, cipher)
    conversations = ConversationStore(database, cipher)
    profiles.put_profile("sam", UserProfile.model_validate({**SAMPLE_PROFILE, "name": "Zebedee"}))
    profiles.add_note("sam", "Rent is due on the first")
    conversations.append_turn("sam", "c1", "I owe 98765 on my card", "Let's look at that.")

    with database.connect() as connection:
        connection.execute("PRAGMA wal_checkpoint(FULL)")
        dumped = "\n".join(connection.iterdump())
    for secret in ("Zebedee", "Rent is due", "98765", "Let's look"):
        assert secret not in dumped
    assert profiles.get_profile("sam").name == "Zebedee"
    assert conversations.get("sam", "c1")["messages"][0]["content"] == "I owe 98765 on my card"


# --- Conversations ---------------------------------------------------------------


def test_conversation_ownership_is_enforced(users: UserStore, database: Database) -> None:
    conversations = ConversationStore(database)
    conversations.append_turn("sam", "c1", "hi", "hello")

    with pytest.raises(ConversationNotFoundError):
        conversations.get("alex", "c1")
    with pytest.raises(ConversationNotFoundError):
        conversations.append_turn("alex", "c1", "sneaky", "no")
    with pytest.raises(ConversationNotFoundError):
        conversations.delete("alex", "c1")
    assert conversations.list("alex") == []
    assert len(conversations.get("sam", "c1")["messages"]) == 2


def test_concurrent_turns_are_never_lost(users: UserStore, database: Database) -> None:
    conversations = ConversationStore(database)
    conversations.append_turn("sam", "shared", "q-seed", "a-seed")
    errors: list[Exception] = []

    def worker(number: int) -> None:
        try:
            for turn in range(5):
                conversations.append_turn("sam", "shared", f"q{number}-{turn}", f"a{number}-{turn}")
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    messages = conversations.get("sam", "shared")["messages"]
    assert len(messages) == 2 + 8 * 5 * 2
    # Each turn's question is immediately followed by its own answer.
    for question, answer in zip(messages[::2], messages[1::2], strict=True):
        assert question["role"] == "user"
        assert answer["role"] == "assistant"
        assert answer["content"] == question["content"].replace("q", "a", 1)


def test_conversation_message_limit(users: UserStore, database: Database) -> None:
    conversations = ConversationStore(database)
    for _ in range(MAX_MESSAGES_PER_CONVERSATION // 2):
        conversations.append_turn("sam", "long", "q", "a")

    with pytest.raises(ConversationFullError):
        conversations.append_turn("sam", "long", "q", "a")


def test_list_orders_newest_first_with_last_message(users: UserStore, database: Database) -> None:
    conversations = ConversationStore(database)
    conversations.append_turn("sam", "older", "q1", "a1")
    conversations.append_turn("sam", "newer", "q2", "a2")

    listed = conversations.list("sam")

    assert [item["conversation_id"] for item in listed] == ["newer", "older"]
    assert listed[0]["last_message"] == "a2"
    assert listed[0]["message_count"] == 2


# --- Legacy JSON import ------------------------------------------------------------


def test_legacy_json_import_is_non_destructive_and_idempotent(tmp_path: Path) -> None:
    profiles_dir = tmp_path / "profiles"
    conversations_dir = tmp_path / "conversations" / "alex"
    profiles_dir.mkdir()
    conversations_dir.mkdir(parents=True)
    legacy_profile = profiles_dir / "alex.json"
    legacy_profile.write_text(
        json.dumps(
            {"profile": SAMPLE_PROFILE, "notes": [{"note_id": "n1", "content": "Likes lists"}]}
        ),
        encoding="utf-8",
    )
    (profiles_dir / "broken.json").write_text("{not json", encoding="utf-8")
    (conversations_dir / "c1.json").write_text(
        json.dumps(
            {
                "conversation_id": "c1",
                "user_id": "alex",
                "created_at": "2026-10-01T00:00:00+00:00",
                "updated_at": "2026-10-01T00:00:00+00:00",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            }
        ),
        encoding="utf-8",
    )
    original = legacy_profile.read_bytes()
    settings = make_settings(tmp_path)

    first = import_json(settings, profiles_dir, tmp_path / "conversations", dry_run=False)
    second = import_json(settings, profiles_dir, tmp_path / "conversations", dry_run=False)

    assert first == {"users": 1, "profiles": 1, "conversations": 1, "skipped": 1}
    assert second["users"] == 0
    assert second["conversations"] == 0
    assert legacy_profile.read_bytes() == original
    database = Database(settings.resolve(settings.database_path))
    assert ProfileStore(database).get_notes("alex")[0].content == "Likes lists"
    assert len(ConversationStore(database).get("alex", "c1")["messages"]) == 2
    # Imported accounts can't log in until a password is set.
    assert not UserStore(database, timedelta(hours=1)).authenticate("alex", "")


def test_legacy_import_dry_run_writes_nothing(tmp_path: Path) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "alex.json").write_text(
        json.dumps({"profile": SAMPLE_PROFILE, "notes": []}), encoding="utf-8"
    )
    settings = make_settings(tmp_path)

    counts = import_json(settings, profiles_dir, tmp_path / "none", dry_run=True)

    assert counts["profiles"] == 1
    assert not UserStore(Database(settings.resolve(settings.database_path)), timedelta(1)).exists(
        "alex"
    )
