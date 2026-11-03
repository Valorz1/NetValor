"""Administrative commands for local NetValor installations.

    python -m app.cli generate-key
    python -m app.cli create-user alex
    python -m app.cli set-password alex
    python -m app.cli import-json [--dry-run]

Passwords are read with getpass, or from NETVALOR_PASSWORD for scripted use.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.config import Settings, get_settings
from app.conversations import ConversationStore
from app.db import Database
from app.memory import MemoryNote, ProfileStore, UserProfile, validate_user_id
from app.security import MIN_PASSWORD_LENGTH, build_cipher, generate_encryption_key
from app.users import UserExistsError, UserStore


def _stores(settings: Settings) -> tuple[UserStore, ProfileStore, ConversationStore]:
    database = Database(settings.resolve(settings.database_path))
    database.initialise()
    keys = (
        settings.data_encryption_keys.get_secret_value() if settings.data_encryption_keys else None
    )
    cipher = build_cipher(keys)
    return (
        UserStore(database, timedelta(hours=settings.session_ttl_hours)),
        ProfileStore(database, cipher),
        ConversationStore(database, cipher),
    )


def _read_password() -> str:
    password = os.environ.get("NETVALOR_PASSWORD") or getpass.getpass("New password: ")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise SystemExit(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if not os.environ.get("NETVALOR_PASSWORD") and getpass.getpass("Repeat: ") != password:
        raise SystemExit("Passwords did not match.")
    return password


def import_json(
    settings: Settings, profiles_dir: Path, conversations_dir: Path, dry_run: bool
) -> dict[str, int]:
    """Copy legacy JSON profiles, notes and conversations into the database.

    Source files are never modified or deleted. Existing database rows win,
    so the import can be re-run safely. Imported accounts have no password
    until one is set with `set-password`.
    """
    users, profiles, conversations = _stores(settings)
    counts = {"users": 0, "profiles": 0, "conversations": 0, "skipped": 0}
    for path in sorted(profiles_dir.glob("*.json")):
        user_id = path.stem.lower()
        try:
            validate_user_id(user_id)
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            profile = UserProfile.model_validate(data["profile"])
            notes = [MemoryNote.model_validate(note) for note in data.get("notes", [])]
        except (ValueError, KeyError, ValidationError) as exc:
            print(f"Skipping {path.name}: {type(exc).__name__}", file=sys.stderr)
            counts["skipped"] += 1
            continue
        if dry_run:
            counts["profiles"] += 1
            continue
        if not users.exists(user_id):
            users.create_user(user_id, None)
            counts["users"] += 1
        profiles.import_record(user_id, profile, notes)
        counts["profiles"] += 1
    if conversations_dir.is_dir():
        for user_dir in sorted(p for p in conversations_dir.iterdir() if p.is_dir()):
            user_id = user_dir.name.lower()
            if dry_run or not users.exists(user_id):
                continue
            for path in sorted(user_dir.glob("*.json")):
                try:
                    conversation = json.loads(path.read_text(encoding="utf-8"))
                    if conversations.import_conversation(user_id, conversation):
                        counts["conversations"] += 1
                except (ValueError, KeyError, TypeError) as exc:
                    print(f"Skipping {path}: {type(exc).__name__}", file=sys.stderr)
                    counts["skipped"] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("generate-key", help="Print a new DATA_ENCRYPTION_KEYS value.")
    create = commands.add_parser("create-user", help="Create a user with a password.")
    create.add_argument("user_id")
    reset = commands.add_parser("set-password", help="Set a password and revoke sessions.")
    reset.add_argument("user_id")
    importer = commands.add_parser("import-json", help="Import legacy JSON data.")
    importer.add_argument("--profiles-dir", type=Path, default=Path("data/profiles"))
    importer.add_argument("--conversations-dir", type=Path, default=Path("data/conversations"))
    importer.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "generate-key":
        print(generate_encryption_key())
        return 0
    settings = get_settings()
    users, _, _ = _stores(settings)
    if args.command == "create-user":
        validate_user_id(args.user_id)
        try:
            users.create_user(args.user_id, _read_password())
        except UserExistsError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Created user {args.user_id}.")
    elif args.command == "set-password":
        users.set_password(args.user_id, _read_password())
        print(f"Password updated for {args.user_id}; existing sessions were revoked.")
    elif args.command == "import-json":
        counts = import_json(
            settings,
            settings.resolve(args.profiles_dir),
            settings.resolve(args.conversations_dir),
            args.dry_run,
        )
        prefix = "Would import" if args.dry_run else "Imported"
        print(
            f"{prefix} {counts['profiles']} profiles ({counts['users']} new users) and "
            f"{counts['conversations']} conversations; skipped {counts['skipped']} files."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
