"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    tavily_api_key: str
    request_timeout: float = 45.0
    llm_timeout: float = 240.0
    max_tool_rounds: int = 12

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
            tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", "45")),
            llm_timeout=float(os.getenv("LLM_TIMEOUT", "240")),
            max_tool_rounds=int(os.getenv("MAX_TOOL_ROUNDS", "12")),
        )

    def validate(self) -> None:
        if not self.llm_model:
            raise ValueError("LLM_MODEL is empty. Set it in your .env file.")
        if not self.llm_base_url.startswith(("http://", "https://")):
            raise ValueError("LLM_BASE_URL must start with http:// or https://.")
        if self.request_timeout <= 0:
            raise ValueError("REQUEST_TIMEOUT must be greater than zero.")
        if self.llm_timeout <= 0:
            raise ValueError("LLM_TIMEOUT must be greater than zero.")
        if not 1 <= self.max_tool_rounds <= 30:
            raise ValueError("MAX_TOOL_ROUNDS must be between 1 and 30.")
