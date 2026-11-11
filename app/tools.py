"""Research tools exposed to the language model through function calling."""

from __future__ import annotations

import ast
import ipaddress
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from typing import Any, Callable
from urllib.parse import quote, urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from app.sources import SourceStore


@dataclass
class ToolContext:
    http: httpx.Client
    sources: SourceStore
    tavily_api_key: str = ""


class ResearchTool:
    name: str
    description: str
    parameters: dict[str, Any]

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        raise NotImplementedError

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _required_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-empty string.")
    return value.strip()


def _clean_text(value: str, limit: int = 2500) -> str:
    return re.sub(r"\s+", " ", value or "").strip()[:limit]


class TavilySearchTool(ResearchTool):
    name = "web_search"
    description = (
        "Search the public web for recent or general information. Use this to discover "
        "independent sources; inspect promising results with fetch_webpage when more detail is needed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A focused web search query."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        query = _required_string(arguments, "query")
        if not context.tavily_api_key:
            raise ValueError("web_search requires TAVILY_API_KEY. Wikipedia and arXiv search are available without it.")
        max_results = max(1, min(int(arguments.get("max_results", 5)), 8))
        response = context.http.post(
            "https://api.tavily.com/search",
            json={
                "api_key": context.tavily_api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": max_results,
                "include_answer": False,
            },
        )
        response.raise_for_status()
        payload = response.json()

        results = []
        for item in payload.get("results", []):
            url = item.get("url")
            if not url:
                continue
            source = context.sources.add(
                title=item.get("title") or url,
                url=url,
                source_type="web",
                published_date=item.get("published_date"),
                snippet=_clean_text(item.get("content", ""), 800),
            )
            results.append(
                {
                    "source_id": source.source_id,
                    "title": source.title,
                    "url": source.url,
                    "published_date": source.published_date,
                    "snippet": source.snippet,
                }
            )
        return {"query": query, "results": results}


class WikipediaSearchTool(ResearchTool):
    name = "wikipedia_search"
    description = "Search Wikipedia for introductory context and useful article links. Do not treat it as the only source for important claims."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Topic or phrase to search on Wikipedia."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        query = _required_string(arguments, "query")
        limit = max(1, min(int(arguments.get("max_results", 5)), 8))
        response = context.http.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "format": "json",
            },
            headers={"User-Agent": "ResearchAgentTraining/1.0 (educational project)"},
        )
        response.raise_for_status()
        payload = response.json()

        results = []
        for item in payload.get("query", {}).get("search", []):
            title = unescape(BeautifulSoup(item.get("title", ""), "html.parser").get_text(" "))
            url = "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))
            snippet = _clean_text(BeautifulSoup(item.get("snippet", ""), "html.parser").get_text(" "), 700)
            source = context.sources.add(
                title=title,
                url=url,
                source_type="wikipedia",
                snippet=snippet,
            )
            results.append({"source_id": source.source_id, "title": title, "url": url, "snippet": snippet})
        return {"query": query, "results": results}


class ArxivSearchTool(ResearchTool):
    name = "arxiv_search"
    description = "Search arXiv for academic preprints. Useful for technical, scientific, and machine-learning research; distinguish preprints from peer-reviewed work."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A concise arXiv search query."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        query = _required_string(arguments, "query")
        limit = max(1, min(int(arguments.get("max_results", 5)), 8))
        response = context.http.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}

        results = []
        for entry in root.findall("atom:entry", namespace):
            title = _clean_text(entry.findtext("atom:title", default="Untitled", namespaces=namespace), 250)
            summary = _clean_text(entry.findtext("atom:summary", default="", namespaces=namespace), 900)
            link = next(
                (
                    node.attrib.get("href", "")
                    for node in entry.findall("atom:link", namespace)
                    if node.attrib.get("rel") == "alternate"
                ),
                "",
            )
            if not link:
                continue
            published = entry.findtext("atom:published", default="", namespaces=namespace)[:10] or None
            source = context.sources.add(
                title=title,
                url=link,
                source_type="arxiv",
                published_date=published,
                snippet=summary,
            )
            results.append(
                {
                    "source_id": source.source_id,
                    "title": title,
                    "url": link,
                    "published_date": published,
                    "summary": summary,
                }
            )
        return {"query": query, "results": results}


def _validate_public_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http and https URLs can be fetched.")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("The URL must contain a public hostname and no embedded credentials.")
    hostname = parsed.hostname.lower().rstrip(".")
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith((".localhost", ".local")):
        raise ValueError("Local and private network addresses cannot be fetched.")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return
    if not address.is_global:
        raise ValueError("Private, local, and reserved IP addresses cannot be fetched.")


class FetchWebpageTool(ResearchTool):
    name = "fetch_webpage"
    description = "Open a public webpage and extract its readable text. Use a URL returned by a search tool or supplied by the user."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Public http(s) URL to read."},
            "max_chars": {"type": "integer", "minimum": 500, "maximum": 12000, "default": 6000},
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        url = _required_string(arguments, "url")
        max_chars = max(500, min(int(arguments.get("max_chars", 6000)), 12000))
        current_url = url
        body = b""
        response_headers: dict[str, str] = {}

        # Validate every redirect target instead of blindly following redirects.
        for _ in range(4):
            _validate_public_url(current_url)
            with context.http.stream(
                "GET",
                current_url,
                headers={"User-Agent": "ResearchAgentTraining/1.0"},
                follow_redirects=False,
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("The page returned a redirect without a Location header.")
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                response_headers = dict(response.headers)
                chunks = bytearray()
                for chunk in response.iter_bytes():
                    remaining = 1_500_000 - len(chunks)
                    if remaining <= 0:
                        break
                    chunks.extend(chunk[:remaining])
                    if len(chunks) >= 1_500_000:
                        break
                body = bytes(chunks)
                break
        else:
            raise ValueError("The webpage redirected too many times.")

        content_type = response_headers.get("content-type", "").lower()
        if content_type and not any(kind in content_type for kind in ("text/html", "text/plain", "application/xhtml")):
            raise ValueError(f"This page does not look like readable HTML or text ({content_type}).")

        encoding = "utf-8"
        try:
            from email.message import Message

            message = Message()
            message["content-type"] = response_headers.get("content-type", "")
            encoding = message.get_content_charset() or "utf-8"
        except Exception:
            pass
        page_text = body.decode(encoding, errors="replace")
        soup = BeautifulSoup(page_text, "html.parser")
        for element in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
            element.decompose()
        title = _clean_text(soup.title.get_text(" ") if soup.title else current_url, 250)
        text = _clean_text(soup.get_text(" ", strip=True), max_chars)
        if not text:
            raise ValueError("No readable text was found on this page.")

        source = context.sources.add(
            title=title,
            url=current_url,
            source_type="webpage",
            snippet=text[:700],
        )
        return {
            "source_id": source.source_id,
            "title": title,
            "url": current_url,
            "text": text,
            "truncated": len(text) >= max_chars,
        }


class CurrentDateTimeTool(ResearchTool):
    name = "get_current_datetime"
    description = (
        "Get the current date and local time from the machine running this agent. "
        "Always use this tool for questions about today's date, the current date, "
        "the current time, or relative dates such as tomorrow and yesterday."
    )
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        now = datetime.now().astimezone()
        return {
            "date": now.strftime("%A, %d %B %Y"),
            "time": now.strftime("%H:%M:%S"),
            "timezone": now.tzname() or "Local time",
            "utc_offset": now.strftime("%z"),
            "iso_datetime": now.isoformat(timespec="seconds"),
        }


class CalculatorTool(ResearchTool):
    name = "calculate"
    description = "Evaluate a basic arithmetic expression when a research answer needs a reproducible calculation. Supports +, -, *, /, //, %, **, and parentheses."
    parameters = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "Arithmetic expression, for example (1200 - 900) / 900 * 100."}},
        "required": ["expression"],
        "additionalProperties": False,
    }

    _operators: dict[type[ast.AST], Callable[[float, float], float]] = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a**b,
    }

    def _evaluate(self, node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return self._evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in self._operators:
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent magnitude is limited to 10.")
            result = self._operators[type(node.op)](left, right)
            if abs(result) > 1e15:
                raise ValueError("The result is too large.")
            return result
        raise ValueError("Only basic arithmetic is allowed; variables and functions are not supported.")

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        expression = _required_string(arguments, "expression")
        if len(expression) > 120:
            raise ValueError("Expression is too long.")
        try:
            parsed = ast.parse(expression, mode="eval")
            result = self._evaluate(parsed)
        except (SyntaxError, ZeroDivisionError, OverflowError) as exc:
            raise ValueError(f"Could not calculate expression: {exc}") from exc
        return {"expression": expression, "result": result}


class ToolRegistry:
    def __init__(self, context: ToolContext) -> None:
        self.context = context
        self._tools: dict[str, ResearchTool] = {}

    def register(self, tool: ResearchTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"A tool named '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            if "_invalid_arguments" in arguments:
                raise ValueError("Tool arguments were not valid JSON.")
            return tool.execute(arguments, self.context)
        except (ValueError, TypeError, httpx.HTTPError, json.JSONDecodeError, ET.ParseError) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}


def build_default_registry(context: ToolContext) -> ToolRegistry:
    registry = ToolRegistry(context)
    for tool in (
        TavilySearchTool(),
        WikipediaSearchTool(),
        ArxivSearchTool(),
        FetchWebpageTool(),
        CalculatorTool(),
        CurrentDateTimeTool(),
    ):
        registry.register(tool)
    return registry
