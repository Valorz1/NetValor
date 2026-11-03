"""Password hashing, session tokens and at-rest encryption of personal data.

* Passwords use scrypt from the standard library (an OWASP-recommended
  memory-hard KDF) with a per-password random salt.
* Session tokens are 256-bit random values. Only their SHA-256 digest is
  stored, so a leaked database cannot be replayed as live sessions.
* Personal financial data (profiles, memory notes and chat messages) can be
  encrypted at rest with Fernet (AES-128-CBC + HMAC-SHA256). Encryption is
  required in production; development may run with a pass-through cipher.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from typing import Protocol

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_MAXMEM = 64 * 1024 * 1024
_SALT_BYTES = 16
_KEY_BYTES = 32
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 128


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str) -> str:
    """Return a self-describing scrypt hash: `scrypt$n$r$p$salt$digest`."""
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        maxmem=_SCRYPT_MAXMEM,
        dklen=_KEY_BYTES,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str | None) -> bool:
    """Check a password in constant time. A missing hash never verifies."""
    if not encoded:
        # Spend comparable time so unknown or disabled accounts are not
        # distinguishable from wrong passwords by response timing.
        hash_password(password)
        return False
    try:
        scheme, n, r, p, salt, expected = encoded.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_unb64(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            maxmem=_SCRYPT_MAXMEM,
            dklen=len(_unb64(expected)),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, _unb64(expected))


def new_session_token() -> str:
    """Create an unguessable bearer token for a client."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """Digest a high-entropy token for storage and lookup."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class Cipher(Protocol):
    """Encrypts and decrypts text values stored in the database."""

    def encrypt(self, plaintext: str) -> str: ...

    def decrypt(self, stored: str) -> str: ...


class DecryptionError(RuntimeError):
    """Raised when stored data cannot be decrypted with the configured keys."""


_ENCRYPTED_PREFIX = "enc:v1:"


class PlaintextCipher:
    """Development-only cipher that stores text unchanged.

    It can still read values written by `FernetCipher`'s format marker so
    that a missing key fails loudly instead of returning ciphertext.
    """

    def encrypt(self, plaintext: str) -> str:
        return plaintext

    def decrypt(self, stored: str) -> str:
        if stored.startswith(_ENCRYPTED_PREFIX):
            raise DecryptionError(
                "Stored data is encrypted but DATA_ENCRYPTION_KEYS is not configured."
            )
        return stored


class FernetCipher:
    """Authenticated symmetric encryption with support for key rotation."""

    def __init__(self, keys: list[str]) -> None:
        if not keys:
            raise ValueError("At least one encryption key is required.")
        try:
            self._fernet = MultiFernet([Fernet(key.strip().encode("ascii")) for key in keys])
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "DATA_ENCRYPTION_KEYS must contain valid Fernet keys "
                "(generate one with `python -m app.cli generate-key`)."
            ) from exc

    def encrypt(self, plaintext: str) -> str:
        token = self._fernet.encrypt(plaintext.encode("utf-8"))
        return _ENCRYPTED_PREFIX + token.decode("ascii")

    def decrypt(self, stored: str) -> str:
        if not stored.startswith(_ENCRYPTED_PREFIX):
            # Data written before encryption was enabled remains readable;
            # it is re-encrypted the next time it is saved.
            return stored
        try:
            return self._fernet.decrypt(stored[len(_ENCRYPTED_PREFIX) :].encode("ascii")).decode(
                "utf-8"
            )
        except InvalidToken as exc:
            raise DecryptionError(
                "Stored data could not be decrypted with the configured keys."
            ) from exc


def build_cipher(keys: str | None) -> Cipher:
    """Create the configured cipher from a comma-separated key list."""
    if not keys or not keys.strip():
        return PlaintextCipher()
    return FernetCipher([key for key in keys.split(",") if key.strip()])


def generate_encryption_key() -> str:
    """Return a new random Fernet key."""
    return Fernet.generate_key().decode("ascii")
