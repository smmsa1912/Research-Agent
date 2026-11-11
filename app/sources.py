"""Source registry used to give research evidence stable citation IDs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Source:
    source_id: str
    title: str
    url: str
    source_type: str = "web"
    published_date: str | None = None
    snippet: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SourceStore:
    """Keeps citation IDs stable for the lifetime of a conversation."""

    def __init__(self) -> None:
        self._sources_by_url: dict[str, Source] = {}
        self._sources_by_id: dict[str, Source] = {}
        self._active_ids: list[str] = []

    def begin_turn(self) -> None:
        self._active_ids = []

    def add(
        self,
        *,
        title: str,
        url: str,
        source_type: str = "web",
        published_date: str | None = None,
        snippet: str | None = None,
    ) -> Source:
        normalized_url = url.strip()
        if not normalized_url:
            raise ValueError("A source URL is required.")

        source = self._sources_by_url.get(normalized_url)
        if source is None:
            source_id = f"S{len(self._sources_by_id) + 1}"
            source = Source(
                source_id=source_id,
                title=title.strip() or normalized_url,
                url=normalized_url,
                source_type=source_type,
                published_date=published_date,
                snippet=snippet,
            )
            self._sources_by_url[normalized_url] = source
            self._sources_by_id[source_id] = source

        if source.source_id not in self._active_ids:
            self._active_ids.append(source.source_id)
        return source

    def get(self, source_id: str) -> Source | None:
        return self._sources_by_id.get(source_id)

    def for_current_turn(self) -> list[Source]:
        return [self._sources_by_id[source_id] for source_id in self._active_ids]

    def all(self) -> list[Source]:
        return list(self._sources_by_id.values())

    def citation_catalog(self) -> list[dict[str, Any]]:
        return [self._sources_by_id[source_id].as_dict() for source_id in self._active_ids]

    def clear(self) -> None:
        self._sources_by_url.clear()
        self._sources_by_id.clear()
        self._active_ids = []
