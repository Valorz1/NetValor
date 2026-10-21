"""Streaming Ollama chat client."""

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx


class LLMError(RuntimeError):
    """Raised when the configured language model cannot answer."""


class OllamaChatClient:
    """Small async interface for streaming chat responses from Ollama."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        num_ctx: int | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        ).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "qwen3.5:9b-q4_K_M")
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("LLM_TEMPERATURE", "0.1"))
        )
        self.num_ctx = (
            num_ctx if num_ctx is not None else int(os.getenv("LLM_NUM_CTX", "8192"))
        )
        self.timeout = timeout

    async def stream_chat(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Yield response content tokens from Ollama's newline-delimited stream."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=body
                ) as response:
                    response.raise_for_status()
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
                            raise LLMError(str(event["error"]))
                        message = event.get("message", {})
                        content = message.get("content", "") if isinstance(message, dict) else ""
                        if content:
                            yield str(content)
                        if event.get("done"):
                            return
        except (httpx.HTTPError, OSError) as exc:
            raise LLMError(f"Could not connect to Ollama: {exc}") from exc
