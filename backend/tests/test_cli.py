"""Admin CLI and evaluation runner entry points."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from app import cli
from app.config import get_settings
from app.db import Database
from app.security import FernetCipher
from app.users import UserStore


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    database_path = tmp_path / "cli.db"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    monkeypatch.setenv("NETVALOR_PASSWORD", "a long cli passphrase")
    get_settings.cache_clear()
    yield database_path
    get_settings.cache_clear()


def test_generate_key_prints_a_usable_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["generate-key"]) == 0

    key = capsys.readouterr().out.strip()
    assert FernetCipher([key]).decrypt(FernetCipher([key]).encrypt("x")) == "x"


def test_create_user_and_set_password(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert cli.main(["create-user", "sam"]) == 0
    users = UserStore(Database(cli_env), timedelta(hours=1))
    assert users.authenticate("sam", "a long cli passphrase")

    monkeypatch.setenv("NETVALOR_PASSWORD", "a different passphrase")
    assert cli.main(["set-password", "sam"]) == 0
    assert users.authenticate("sam", "a different passphrase")

    with pytest.raises(SystemExit, match="not available"):
        cli.main(["create-user", "sam"])


def test_create_user_rejects_short_passwords(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NETVALOR_PASSWORD", "short")

    with pytest.raises(SystemExit, match="at least"):
        cli.main(["create-user", "sam"])


def test_import_json_command(
    cli_env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profiles = tmp_path / "profiles"
    profiles.mkdir()

    assert cli.main(["import-json", "--profiles-dir", str(profiles), "--dry-run"]) == 0
    assert "Would import 0 profiles" in capsys.readouterr().out


def test_eval_runner_writes_a_json_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evals import __main__ as runner

    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["evals", "--case", "missing-evidence", "--json", str(report)])

    with pytest.raises(SystemExit) as exit_info:
        runner.main()

    assert exit_info.value.code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["summary"] == {**payload["summary"], "cases": 1, "passed": 1}
    assert payload["results"][0]["id"] == "missing-evidence"
