"""Parsing and validation for the knowledge-base lesson files."""

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


_FRONT_MATTER = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", re.DOTALL)
_REQUIRED_FIELDS = {
    "id",
    "title",
    "module",
    "order",
    "difficulty",
    "jurisdiction",
    "sources",
    "licence",
    "review_status",
    "last_reviewed",
}


@dataclass(frozen=True)
class Lesson:
    metadata: dict[str, Any]
    content: str
    path: Path


def parse_lesson(text: str, path: Path = Path("<lesson>")) -> Lesson:
    """Parse one Markdown lesson and validate its required front matter."""
    match = _FRONT_MATTER.match(text)
    if match is None:
        raise ValueError(f"{path}: missing YAML front matter.")

    metadata = yaml.safe_load(match.group(1))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: YAML front matter must be a mapping.")
    missing = _REQUIRED_FIELDS - metadata.keys()
    if missing:
        raise ValueError(f"{path}: missing fields: {', '.join(sorted(missing))}.")
    for field in ("id", "title", "module", "difficulty", "jurisdiction", "licence", "review_status"):
        if not isinstance(metadata[field], str) or not metadata[field].strip():
            raise ValueError(f"{path}: '{field}' must be a non-empty string.")
    if not isinstance(metadata["order"], int) or isinstance(metadata["order"], bool):
        raise ValueError(f"{path}: 'order' must be an integer.")
    if not isinstance(metadata["last_reviewed"], date):
        raise ValueError(f"{path}: 'last_reviewed' must be a date.")
    sources = metadata["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"{path}: 'sources' must be a non-empty list.")
    for source in sources:
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("publisher"), str)
            or not isinstance(source.get("url"), str)
        ):
            raise ValueError(f"{path}: each source needs a publisher and URL.")
    content = text[match.end():].strip()
    if not content:
        raise ValueError(f"{path}: lesson content is empty.")
    return Lesson(metadata=metadata, content=content, path=path)


def load_lessons(lessons_dir: Path) -> list[Lesson]:
    """Load and order all Markdown lessons in a directory."""
    paths = sorted(Path(lessons_dir).glob("*.md"))
    lessons = [parse_lesson(path.read_text(encoding="utf-8"), path) for path in paths]
    lessons.sort(key=lambda lesson: (lesson.metadata["order"], lesson.metadata["id"]))
    ids = [lesson.metadata["id"] for lesson in lessons]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{lessons_dir}: lesson IDs must be unique.")
    return lessons


def split_lesson_by_heading(lesson: Lesson) -> list[tuple[str, str]]:
    """Return lesson sections, each paired with its level-two heading."""
    sections: list[tuple[str, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    for line in lesson.content.splitlines():
        if line.startswith("## "):
            if current_heading is not None:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line[3:].strip()
            current_lines = []
        elif current_heading is not None:
            current_lines.append(line)
    if current_heading is not None:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    if not sections:
        raise ValueError(f"{lesson.path}: lesson must contain at least one '##' heading.")
    return sections


_TOKEN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)
_MAX_CHUNK_TOKENS = 450


def chunk_lesson(
    lesson: Lesson,
    target_tokens: int = 350,
    overlap_tokens: int = 50,
    minimum_tokens: int = 250,
) -> list[dict[str, str]]:
    """Split lesson sections into approximate token-sized chunks with heading context."""
    if target_tokens < 1 or overlap_tokens < 0 or minimum_tokens < 1:
        raise ValueError("Chunk sizes must be positive and overlap cannot be negative.")
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be smaller than target_tokens.")
    if minimum_tokens > target_tokens:
        raise ValueError("minimum_tokens cannot exceed target_tokens.")

    sections = split_lesson_by_heading(lesson)
    content = "\n\n".join(
        f"## {heading}\n\n{section}" for heading, section in sections if section
    )
    tokens = list(_TOKEN.finditer(content))
    if not tokens:
        return []
    heading_starts = [
        (len(_TOKEN.findall(content[:match.start()])), match.group(1).strip())
        for match in re.finditer(r"(?m)^## (.+)$", content)
    ]
    chunks: list[dict[str, str]] = []
    start = 0
    while start < len(tokens):
        remaining = len(tokens) - start
        end = len(tokens) if remaining <= _MAX_CHUNK_TOKENS else start + target_tokens
        if end < len(tokens) and len(tokens) - end + overlap_tokens < minimum_tokens:
            end = min(
                end,
                max(
                    start + minimum_tokens,
                    len(tokens) + overlap_tokens - minimum_tokens,
                ),
            )
        body_start = tokens[start].start()
        body_end = tokens[end - 1].end()
        heading_start, heading = max(
            (item for item in heading_starts if item[0] <= start),
            key=lambda item: item[0],
        )
        text = content[body_start:body_end]
        if start > heading_start:
            text = f"## {heading}\n\n{text}"
        chunks.append({"heading": heading, "text": text})
        if end == len(tokens):
            break
        start = max(start + 1, end - overlap_tokens)
    return chunks
