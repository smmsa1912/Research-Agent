from __future__ import annotations

from typing import Any, Sequence

import httpx

from app.agent import ResearchAgent
from app.llm import LLMResponse, ToolCall
from app.sources import SourceStore
from app.tools import ResearchTool, ToolContext, ToolRegistry


class FakeProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> LLMResponse:
        self.requests.append((list(messages), list(tools)))
        return self.responses.pop(0)


class ExampleSearchTool(ResearchTool):
    name = "example_search"
    description = "A tiny test-only search tool."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        source = context.sources.add(
            title="Example source",
            url="https://example.com/research",
            source_type="test",
            snippet="A short piece of source text.",
        )
        return {"source_id": source.source_id, "query": arguments["query"]}


def make_agent(provider: FakeProvider, max_tool_rounds: int = 4):
    sources = SourceStore()
    context = ToolContext(http=httpx.Client(), sources=sources)
    registry = ToolRegistry(context)
    registry.register(ExampleSearchTool())
    agent = ResearchAgent(
        provider=provider,
        tools=registry,
        sources=sources,
        max_tool_rounds=max_tool_rounds,
    )
    return agent, sources, context.http


def test_agent_calls_tool_then_returns_cited_answer() -> None:
    provider = FakeProvider(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        call_id="call-1",
                        name="example_search",
                        arguments={"query": "agentic research"},
                    )
                ]
            ),
            LLMResponse(content="The source describes the research topic. [S1]"),
        ]
    )
    agent, _, http = make_agent(provider)
    try:
        result = agent.ask("Research this topic.")
    finally:
        http.close()

    assert result.tool_calls_made == ["example_search"]
    assert result.tool_rounds == 1
    assert "[S1]" in result.content
    assert "https://example.com/research" in result.content
    assert len(result.sources) == 1


def test_agent_stops_tool_loop_and_requests_synthesis() -> None:
    provider = FakeProvider(
        [
            LLMResponse(tool_calls=[ToolCall("call-1", "example_search", {"query": "one"})]),
            LLMResponse(content="Best effort answer based on the available result."),
        ]
    )
    agent, _, http = make_agent(provider, max_tool_rounds=1)
    try:
        result = agent.ask("Find one fact.")
    finally:
        http.close()

    assert result.stopped_at_limit is True
    assert result.tool_rounds == 1
    assert provider.requests[-1][1] == []
    assert "Best effort answer" in result.content


def test_follow_up_keeps_previous_conversation_messages() -> None:
    provider = FakeProvider(
        [
            LLMResponse(content="First answer."),
            LLMResponse(content="A follow-up using the earlier context."),
        ]
    )
    agent, _, http = make_agent(provider)
    try:
        agent.ask("First question")
        agent.ask("Can you expand on that?")
    finally:
        http.close()

    second_messages = provider.requests[1][0]
    assert any(message.get("content") == "First question" for message in second_messages)
    assert any(message.get("content") == "Can you expand on that?" for message in second_messages)


def test_unknown_tool_returns_actionable_error() -> None:
    provider = FakeProvider(
        [
            LLMResponse(tool_calls=[ToolCall("call-1", "not_a_tool", {})]),
            LLMResponse(content="The requested tool was unavailable."),
        ]
    )
    agent, _, http = make_agent(provider)
    try:
        agent.ask("Use a made-up tool.")
    finally:
        http.close()

    messages = provider.requests[1][0]
    tool_messages = [message for message in messages if message.get("role") == "tool"]
    assert "Unknown tool" in tool_messages[0]["content"]


def test_source_store_reuses_ids_for_the_same_url() -> None:
    sources = SourceStore()
    first = sources.add(title="First title", url="https://example.com/page")
    second = sources.add(title="Updated title", url="https://example.com/page")

    assert first.source_id == second.source_id == "S1"
    assert len(sources.all()) == 1


def test_unknown_citation_is_marked_instead_of_being_published_as_valid() -> None:
    provider = FakeProvider(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall("call-1", "example_search", {"query": "source check"})
                ]
            ),
            LLMResponse(content="A supported statement [S1]. Another claim [S99]."),
        ]
    )
    agent, _, http = make_agent(provider)
    try:
        result = agent.ask("Research and cite this.")
    finally:
        http.close()

    assert "[S1]" in result.content
    assert "[unknown source S99]" in result.content
    assert "https://example.com/research" in result.content


def test_sourced_research_answer_gets_a_report_heading() -> None:
    formatted = ResearchAgent._ensure_report_heading("The evidence points to a useful finding.")
    assert formatted.startswith("# Research Report\n\n## Key Findings")


def test_existing_report_sections_are_preserved() -> None:
    original = "## Executive Summary\nA summary.\n\n## Key Findings\nA finding."
    formatted = ResearchAgent._ensure_report_heading(original)
    assert formatted.startswith("# Research Report")
    assert "## Executive Summary" in formatted
    assert "## Key Findings" in formatted


def test_sourced_answer_without_inline_citations_discloses_the_gap() -> None:
    provider = FakeProvider(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall("call-1", "example_search", {"query": "uncited"})
                ]
            ),
            LLMResponse(content="The research produced a useful finding."),
        ]
    )
    agent, _, http = make_agent(provider)
    try:
        result = agent.ask("Research this topic.")
    finally:
        http.close()

    assert "did not attach inline citations" in result.content
    assert "https://example.com/research" in result.content


def test_sources_from_previous_turn_remain_available_for_follow_up_citations() -> None:
    provider = FakeProvider(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall("call-1", "example_search", {"query": "first turn"})
                ]
            ),
            LLMResponse(content="The first finding is supported. [S1]"),
            LLMResponse(content="The same source supports the follow-up. [S1]"),
        ]
    )
    agent, _, http = make_agent(provider)
    try:
        agent.ask("Research the topic.")
        result = agent.ask("Expand the finding.")
    finally:
        http.close()

    assert "[S1]" in result.content
    assert "https://example.com/research" in result.content
    assert len(result.sources) == 1
