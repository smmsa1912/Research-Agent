"""Provider abstraction and an OpenAI-compatible chat-completions adapter.

The agent depends on the LLMProvider protocol, not on a particular vendor SDK.
Any service exposing an OpenAI-compatible /chat/completions endpoint can be
configured through LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import httpx


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def as_message(self) -> dict[str, Any]:
        """Convert the normalized response back to chat-completions format."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in self.tool_calls
            ]
        return message


class LLMProvider(Protocol):
    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> LLMResponse:
        """Return either a normal assistant response or one/more tool calls."""


class ProviderError(RuntimeError):
    """Raised when the configured language model cannot complete a request."""


class OpenAICompatibleProvider:
    """Thin HTTP adapter for providers with the OpenAI chat-completions shape."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=timeout)

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> LLMResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"

        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:800]
            raise ProviderError(
                f"LLM request failed ({exc.response.status_code}): {body}"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(f"Could not reach the LLM provider: {exc}") from exc

        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("The LLM provider returned an unexpected response.") from exc

        calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            raw_arguments = function.get("arguments", "{}")
            try:
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                if not isinstance(arguments, dict):
                    arguments = {"_invalid_arguments": arguments}
            except (json.JSONDecodeError, TypeError):
                arguments = {"_invalid_arguments": raw_arguments}

            calls.append(
                ToolCall(
                    call_id=str(item.get("id", "tool-call")),
                    name=str(function.get("name", "")),
                    arguments=arguments,
                )
            )

        content = message.get("content")
        if content is not None and not isinstance(content, str):
            # Some compatible endpoints return content blocks rather than text.
            content = str(content)
        return LLMResponse(content=content, tool_calls=calls)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()
