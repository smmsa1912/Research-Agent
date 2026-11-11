from __future__ import annotations

import httpx

from app.llm import OpenAICompatibleProvider


def test_openai_compatible_provider_parses_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-123",
                                    "type": "function",
                                    "function": {
                                        "name": "wikipedia_search",
                                        "arguments": '{"query":"tool calling"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        base_url="https://llm.example/v1",
        api_key="test-key",
        model="test-model",
        client=client,
    )
    try:
        result = provider.complete([{"role": "user", "content": "Search"}], [])
    finally:
        client.close()

    assert result.content is None
    assert result.tool_calls[0].name == "wikipedia_search"
    assert result.tool_calls[0].arguments == {"query": "tool calling"}


def test_provider_sends_native_tool_definitions_and_auto_tool_choice() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Done."}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        base_url="https://llm.example/v1",
        api_key="test-key",
        model="test-model",
        client=client,
    )
    schema = {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Read the current date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    try:
        provider.complete([{"role": "user", "content": "What date is it?"}], [schema])
    finally:
        client.close()

    assert captured["tools"] == [schema]
    assert captured["tool_choice"] == "auto"


def test_provider_reports_a_clear_timeout_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("model response timed out", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=1.0)
    provider = OpenAICompatibleProvider(
        base_url="https://llm.example/v1",
        api_key="test-key",
        model="test-model",
        client=client,
    )
    from app.llm import ProviderError

    try:
        try:
            provider.complete([{"role": "user", "content": "Research"}], [])
            assert False, "Expected ProviderError"
        except ProviderError as exc:
            assert "LLM_TIMEOUT" in str(exc)
            assert "try a smaller research query" in str(exc)
    finally:
        client.close()
