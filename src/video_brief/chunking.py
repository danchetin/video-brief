"""Group transcript segments into chunks that fit an LLM context budget."""

from __future__ import annotations

from .subtitles import Segment, format_timestamp


def chunk_segments(segments: list[Segment], max_chars: int = 12_000) -> list[list[Segment]]:
    """Split segments into consecutive chunks of at most ``max_chars`` characters.

    Segment boundaries are never split, so every chunk starts and ends on a
    timestamped sentence. A single segment longer than ``max_chars`` gets its own chunk.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    chunks: list[list[Segment]] = []
    current: list[Segment] = []
    size = 0
    for segment in segments:
        length = len(segment.text) + 8  # room for the "[MM:SS] " prefix
        if current and size + length > max_chars:
            chunks.append(current)
            current, size = [], 0
        current.append(segment)
        size += length
    if current:
        chunks.append(current)
    return chunks


def to_prompt_text(segments: list[Segment]) -> str:
    """Render segments as ``[MM:SS] text`` lines for an LLM prompt."""
    return "\n".join(f"[{format_timestamp(s.start)}] {s.text}" for s in segments)
