"""Parse SRT / WebVTT subtitle files into timed text segments."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ARROW = re.compile(r"(\S+)\s+-->\s+(\S+)")
_TIMESTAMP = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[,.](\d{3})")
_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class Segment:
    """A piece of speech: start/end in seconds and the spoken text."""

    start: float
    end: float
    text: str


def parse_timestamp(value: str) -> float:
    """Parse ``HH:MM:SS,mmm`` / ``HH:MM:SS.mmm`` / ``MM:SS.mmm`` into seconds."""
    match = _TIMESTAMP.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"not a subtitle timestamp: {value!r}")
    hours, minutes, seconds, millis = match.groups()
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``MM:SS`` (or ``H:MM:SS`` from one hour up)."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_subtitles(text: str, *, dedupe_rolling: bool = False) -> list[Segment]:
    """Parse SRT or WebVTT text.

    ``dedupe_rolling`` handles YouTube auto-captions, where every cue repeats the
    previous cue's last line while the next words scroll in. A line that already
    appeared in the previous cue is dropped, and cues left empty are skipped.
    """
    segments: list[Segment] = []
    previous_lines: set[str] = set()

    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = block.split("\n")
        header = next((i for i, line in enumerate(lines) if _ARROW.search(line)), None)
        if header is None:
            continue  # WEBVTT header, NOTE blocks, bare numbers
        arrow = _ARROW.search(lines[header])
        assert arrow is not None
        try:
            start = parse_timestamp(arrow.group(1))
            end = parse_timestamp(arrow.group(2))
        except ValueError:
            continue

        cleaned = [_SPACES.sub(" ", _TAG.sub("", line)).strip() for line in lines[header + 1 :]]
        cleaned = [line for line in cleaned if line]

        if dedupe_rolling:
            fresh = [line for line in cleaned if line not in previous_lines]
            previous_lines = set(cleaned)
            cleaned = fresh

        if cleaned:
            segments.append(Segment(start, end, " ".join(cleaned)))

    return segments
