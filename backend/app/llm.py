"""Streaming Ollama chat client."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(120.0, connect=5.0)


class LLMError(RuntimeError):
    """Raised when the configured language model cannot answer.

    Messages are for logs only; they are never shown to API clients.
    """


class OllamaChatClient:
    """Stream chat responses from Ollama over a shared async HTTP client."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3.5:9b-q4_K_M",
        temperature: float = 0.1,
        num_ctx: int = 8192,
        num_predict: int = 700,
        http_client: httpx.AsyncClient | None = None,
        timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self._client = http_client
        self._timeout = timeout

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Yield content tokens from Ollama's newline-delimited JSON stream."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        owned = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self._timeout)
        try:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=body) as response:
                if response.status_code >= 400:
                    raise LLMError(f"Ollama returned HTTP {response.status_code}.")
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise LLMError("Ollama returned malformed chat data.") from exc
                    if not isinstance(event, dict):
                        raise LLMError("Ollama returned an invalid chat event.")
                    if event.get("error"):
                        raise LLMError(f"Ollama error: {event['error']}")
                    message = event.get("message", {})
                    content = message.get("content", "") if isinstance(message, dict) else ""
                    if content:
                        yield str(content)
                    if event.get("done"):
                        return
                raise LLMError("Ollama closed the stream before finishing.")
        except httpx.TimeoutException as exc:
            raise LLMError("Timed out waiting for Ollama.") from exc
        except (httpx.HTTPError, OSError) as exc:
            raise LLMError(f"Could not connect to Ollama: {type(exc).__name__}") from exc
        finally:
            if owned:
                await client.aclose()
