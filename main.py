"""Run the research agent as an interactive terminal application."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

from app.agent import ResearchAgent
from app.config import Settings
from app.llm import OpenAICompatibleProvider
from app.sources import SourceStore
from app.tools import ToolContext, build_default_registry


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"


def save_report(markdown: str) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"research_report_{timestamp}.md"
    path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    return path


def print_help() -> None:
    print(
        "\nCommands\n"
        "  :help   Show this help message\n"
        "  :reset  Clear the current conversation and its citations\n"
        "  :quit   Exit the application\n"
        "\nEnter a research question to start. Follow-up questions share the same context.\n"
    )


def main() -> None:
    load_dotenv(ROOT / ".env")
    settings = Settings.from_env()
    settings.validate()

    if not settings.llm_api_key and "localhost" not in settings.llm_base_url and "127.0.0.1" not in settings.llm_base_url:
        print("Warning: LLM_API_KEY is empty. This is usually only valid for local model servers.\n")

    provider = OpenAICompatibleProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=settings.request_timeout,
    )
    sources = SourceStore()
    with httpx.Client(timeout=settings.request_timeout) as http:
        context = ToolContext(
            http=http,
            sources=sources,
            tavily_api_key=settings.tavily_api_key,
        )
        registry = build_default_registry(context)
        agent = ResearchAgent(
            provider=provider,
            tools=registry,
            sources=sources,
            max_tool_rounds=settings.max_tool_rounds,
        )

        print("Research Agent")
        print(f"Model: {settings.llm_model}")
        print("Provider endpoint: configured via LLM_BASE_URL")
        print("Type :help for commands or :quit to exit.\n")

        try:
            while True:
                try:
                    prompt = input("you> ").strip()
                except EOFError:
                    print("\nGoodbye.")
                    break

                if not prompt:
                    continue
                command = prompt.lower()
                if command in {":quit", ":exit"}:
                    print("Goodbye.")
                    break
                if command == ":help":
                    print_help()
                    continue
                if command == ":reset":
                    agent.reset()
                    print("Conversation and citations cleared.\n")
                    continue

                try:
                    result = agent.ask(prompt)
                except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                    print(f"\nAgent error: {exc}\n")
                    continue

                print(f"\nagent>\n{result.content}\n")
                if result.tool_calls_made:
                    unique_tools = ", ".join(dict.fromkeys(result.tool_calls_made))
                    print(f"Tools used: {unique_tools}")
                else:
                    print("Tools used: none (the model answered without invoking a tool).")
                if result.stopped_at_limit:
                    print("Note: the tool-call limit was reached; this is a best-effort synthesis.")
                if result.sources:
                    report_path = save_report(result.content)
                    print(f"Report saved to: {report_path.relative_to(ROOT)}\n")

        finally:
            provider.close()


if __name__ == "__main__":
    main()
