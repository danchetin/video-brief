"""Render transcripts as readable Markdown."""

from __future__ import annotations

from .subtitles import Segment, format_timestamp


def paragraphs(segments: list[Segment], window_seconds: float = 30.0) -> list[tuple[float, str]]:
    """Merge segments into paragraphs, starting a new one every ``window_seconds``."""
    result: list[tuple[float, str]] = []
    start = 0.0
    words: list[str] = []
    for segment in segments:
        if words and segment.start - start >= window_seconds:
            result.append((start, " ".join(words)))
            words = []
        if not words:
            start = segment.start
        words.append(segment.text)
    if words:
        result.append((start, " ".join(words)))
    return result


def transcript_markdown(segments: list[Segment], *, title: str, source: str) -> str:
    """Full transcript with a ``[MM:SS]`` marker at the start of each paragraph."""
    lines = [f"# {title}", "", f"Source: {source}", ""]
    for start, text in paragraphs(segments):
        lines.append(f"**[{format_timestamp(start)}]** {text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
