from __future__ import annotations

import httpx
import pytest

from app.sources import SourceStore
from app.tools import CalculatorTool, CurrentDateTimeTool, ToolContext, _validate_public_url


def test_calculator_handles_arithmetic_without_eval() -> None:
    context = ToolContext(http=httpx.Client(), sources=SourceStore())
    try:
        result = CalculatorTool().execute(
            {"expression": "(1200 - 900) / 900 * 100"},
            context,
        )
    finally:
        context.http.close()

    assert result["result"] == pytest.approx(33.3333333333)


def test_calculator_rejects_function_calls() -> None:
    context = ToolContext(http=httpx.Client(), sources=SourceStore())
    try:
        with pytest.raises(ValueError, match="Only basic arithmetic"):
            CalculatorTool().execute({"expression": "__import__('os').system('whoami')"}, context)
    finally:
        context.http.close()


def test_page_reader_rejects_local_addresses() -> None:
    with pytest.raises(ValueError, match="Private, local"):
        _validate_public_url("http://127.0.0.1/admin")


def test_current_datetime_tool_returns_local_date_and_timezone() -> None:
    context = ToolContext(http=httpx.Client(), sources=SourceStore())
    try:
        result = CurrentDateTimeTool().execute({}, context)
    finally:
        context.http.close()

    assert result["date"]
    assert result["time"]
    assert result["timezone"]
    assert result["utc_offset"]
    assert "T" in result["iso_datetime"]


def test_default_registry_exposes_at_least_three_external_research_tools() -> None:
    from app.tools import build_default_registry

    context = ToolContext(http=httpx.Client(), sources=SourceStore())
    try:
        registry = build_default_registry(context)
        names = {schema["function"]["name"] for schema in registry.schemas()}
    finally:
        context.http.close()

    assert {"web_search", "wikipedia_search", "arxiv_search", "fetch_webpage"}.issubset(names)
    assert "get_current_datetime" in names
    assert "calculate" in names
