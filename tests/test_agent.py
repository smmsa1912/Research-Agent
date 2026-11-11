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
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


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


def test_agent_reports_progress_for_model_and_tool_steps() -> None:
    provider = FakeProvider(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall("call-1", "example_search", {"query": "progress"})
                ]
            ),
            LLMResponse(content="The result is useful. [S1]"),
        ]
    )
    agent, _, http = make_agent(provider)
    updates: list[str] = []
    try:
        agent.ask("Research and report progress.", on_progress=updates.append)
    finally:
        http.close()

    assert any("Thinking" in update for update in updates)
    assert "Running tool: example_search" in updates
    assert any("final response" in update for update in updates)


def test_agent_can_finish_early_without_using_the_full_round_budget() -> None:
    provider = FakeProvider([LLMResponse(content="A concise answer without searching.")])
    agent, _, http = make_agent(provider, max_tool_rounds=12)
    try:
        result = agent.ask("What is a software agent?")
    finally:
        http.close()

    assert result.tool_rounds == 0
    assert result.stop_reason == "model_finished"
    assert result.stopped_at_limit is False


def test_agent_can_continue_beyond_seven_rounds_until_model_finishes() -> None:
    responses = [
        LLMResponse(
            tool_calls=[
                ToolCall(f"call-{index}", "example_search", {"query": f"topic {index}"})
            ]
        )
        for index in range(1, 9)
    ]
    responses.append(LLMResponse(content="A synthesized answer after eight research rounds. [S1]"))
    provider = FakeProvider(responses)
    agent, _, http = make_agent(provider, max_tool_rounds=12)
    try:
        result = agent.ask("Research a broad topic.")
    finally:
        http.close()

    assert result.tool_rounds == 8
    assert result.stopped_at_limit is False
    assert result.stop_reason == "model_finished"


def test_tool_limit_synthesis_uses_compact_prompt_not_full_tool_transcript() -> None:
    provider = FakeProvider(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall("call-1", "example_search", {"query": "compact evidence"})
                ]
            ),
            LLMResponse(content="A best-effort synthesis. [S1]"),
        ]
    )
    agent, _, http = make_agent(provider, max_tool_rounds=1)
    try:
        result = agent.ask("Research this and report.")
    finally:
        http.close()

    synthesis_messages, synthesis_tools = provider.requests[-1]
    assert synthesis_tools == []
    assert len(synthesis_messages) == 2
    assert "Collected source evidence" in synthesis_messages[-1]["content"]
    assert "call-1" not in synthesis_messages[-1]["content"]
    assert result.stopped_at_limit is True


def test_agent_recovers_with_compact_synthesis_after_late_provider_timeout() -> None:
    from app.llm import ProviderTimeoutError

    provider = FakeProvider(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall("call-1", "example_search", {"query": "timeout recovery"})
                ]
            ),
            ProviderTimeoutError("fake timeout"),
            LLMResponse(content="Recovered summary based on collected evidence. [S1]"),
        ]
    )
    agent, _, http = make_agent(provider, max_tool_rounds=12)
    try:
        result = agent.ask("Research and cite this topic.")
    finally:
        http.close()

    assert result.degraded is True
    assert result.stop_reason == "recovered_after_provider_timeout"
    assert "Recovered summary" in result.content
    assert "https://example.com/research" in result.content


def test_agent_returns_evidence_report_if_compact_synthesis_also_times_out() -> None:
    from app.llm import ProviderTimeoutError

    provider = FakeProvider(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall("call-1", "example_search", {"query": "timeout fallback"})
                ]
            ),
            ProviderTimeoutError("fake timeout during finalization"),
        ]
    )
    agent, _, http = make_agent(provider, max_tool_rounds=1)
    try:
        result = agent.ask("Research this topic.")
    finally:
        http.close()

    assert result.degraded is True
    assert result.stopped_at_limit is True
    assert result.stop_reason == "tool_round_limit_and_synthesis_timeout"
    assert "not a completed comparative analysis" in result.content
    assert "https://example.com/research" in result.content


def test_bounded_messages_compacts_old_tool_replies_and_keeps_recent_ones() -> None:
    provider = FakeProvider([LLMResponse(content="done")])
    agent, sources, http = make_agent(provider)
    try:
        sources.add(
            title="Useful source",
            url="https://example.com/useful",
            snippet="Evidence snippet for the source index.",
        )
        agent.messages.append({"role": "user", "content": "A large research task"})
        for index in range(10):
            agent.messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [{"id": f"call-{index}", "type": "function", "function": {"name": "example_search", "arguments": "{}"}}],
                    "content": None,
                }
            )
            agent.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call-{index}",
                    "name": "example_search",
                    "content": '{"source_id":"S1","data":"' + ("x" * 6000) + '"}',
                }
            )
        bounded = agent._bounded_messages()
    finally:
        http.close()

    tool_messages = [message for message in bounded if message.get("role") == "tool"]
    assert len(tool_messages) == 10
    assert len(tool_messages[0]["content"]) < 600
    assert len(tool_messages[-1]["content"]) > 5_000
    assert any("Previously collected source index" in str(m.get("content")) for m in bounded)
    assert sum(len(str(m.get("content") or "")) for m in bounded) < agent.MAX_REQUEST_CONTEXT_CHARS
