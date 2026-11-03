from __future__ import annotations

import pytest

from app import security
from app.config import Settings
from app.security import (
    DecryptionError,
    FernetCipher,
    PlaintextCipher,
    build_cipher,
    generate_encryption_key,
    hash_password,
    hash_token,
    new_session_token,
    verify_password,
)


def test_password_hash_round_trip_and_salting() -> None:
    first = hash_password("a long passphrase")
    second = hash_password("a long passphrase")

    assert first != second  # random salt
    assert first.startswith("scrypt$")
    assert verify_password("a long passphrase", first)
    assert not verify_password("a long passphrasE", first)


def test_production_cost_parameters_are_recorded_in_the_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(security, "_SCRYPT_N", 2**15)
    encoded = hash_password("a long passphrase")

    assert encoded.split("$")[1:4] == ["32768", "8", "1"]
    assert verify_password("a long passphrase", encoded)


@pytest.mark.parametrize("stored", [None, "", "md5$abc", "scrypt$bad", "scrypt$1$2$3$!!$!!"])
def test_missing_or_malformed_hashes_never_verify(stored: str | None) -> None:
    assert verify_password("anything at all", stored) is False


def test_session_tokens_are_random_and_stored_hashed() -> None:
    tokens = {new_session_token() for _ in range(50)}

    assert len(tokens) == 50
    token = tokens.pop()
    assert len(token) >= 40
    assert hash_token(token) != token
    assert len(hash_token(token)) == 64


def test_fernet_cipher_round_trip_and_marker() -> None:
    cipher = FernetCipher([generate_encryption_key()])
    stored = cipher.encrypt("I owe £2,000 on my card")

    assert stored.startswith("enc:v1:")
    assert "2,000" not in stored
    assert cipher.decrypt(stored) == "I owe £2,000 on my card"


def test_fernet_cipher_reads_legacy_plaintext() -> None:
    assert FernetCipher([generate_encryption_key()]).decrypt("plain") == "plain"


def test_key_rotation_keeps_old_data_readable() -> None:
    old_key, new_key = generate_encryption_key(), generate_encryption_key()
    stored = FernetCipher([old_key]).encrypt("secret")

    rotated = build_cipher(f"{new_key},{old_key}")

    assert rotated.decrypt(stored) == "secret"
    with pytest.raises(DecryptionError):
        FernetCipher([new_key]).decrypt(stored)


def test_plaintext_cipher_refuses_to_return_ciphertext() -> None:
    stored = FernetCipher([generate_encryption_key()]).encrypt("secret")

    with pytest.raises(DecryptionError, match="DATA_ENCRYPTION_KEYS"):
        PlaintextCipher().decrypt(stored)


def test_invalid_keys_are_rejected_with_guidance() -> None:
    with pytest.raises(ValueError, match="generate-key"):
        build_cipher("not-a-valid-key")
    assert isinstance(build_cipher(None), PlaintextCipher)
    assert isinstance(build_cipher("  "), PlaintextCipher)


# --- Production configuration guard -----------------------------------------


def _production(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": "production",
        "cors_origins": "https://app.netvalor.example",
        "data_encryption_keys": generate_encryption_key(),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_safe_production_configuration_is_accepted() -> None:
    settings = _production()

    assert settings.show_docs is False
    assert settings.allowed_review_statuses == frozenset({"approved"})


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        ({"allow_draft": True}, "ALLOW_DRAFT"),
        ({"seed_demo_user": True}, "SEED_DEMO_USER"),
        ({"cors_origins": "*"}, "CORS_ORIGINS"),
        ({"cors_origins": "http://app.example"}, "https://"),
        ({"data_encryption_keys": None}, "DATA_ENCRYPTION_KEYS"),
        ({"docs_enabled": True}, "DOCS_ENABLED"),
    ],
)
def test_unsafe_production_configuration_is_refused(
    overrides: dict[str, object], problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        _production(**overrides)


def test_development_defaults_are_conservative() -> None:
    settings = Settings(_env_file=None)

    assert settings.allow_draft is False
    assert settings.seed_demo_user is False
    assert settings.cors_origins == ["http://localhost:3000"]
    assert settings.show_docs is True


def test_cors_origins_parse_from_comma_separated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000, http://127.0.0.1:3000")

    assert Settings(_env_file=None).cors_origins == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


def test_invalid_ollama_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="OLLAMA_URL"):
        Settings(_env_file=None, ollama_url="localhost:11434")
