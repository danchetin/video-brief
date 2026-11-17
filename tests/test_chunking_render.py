import pytest

from video_brief.chunking import chunk_segments, to_prompt_text
from video_brief.render import paragraphs, transcript_markdown
from video_brief.subtitles import Segment


def seg(start, text):
    return Segment(start, start + 1, text)


def test_chunks_respect_budget_and_keep_order():
    segments = [seg(i, "x" * 40) for i in range(10)]
    chunks = chunk_segments(segments, max_chars=150)
    assert [s for chunk in chunks for s in chunk] == segments
    assert all(sum(len(s.text) + 8 for s in chunk) <= 150 for chunk in chunks)
    assert len(chunks) > 1


def test_oversized_segment_gets_its_own_chunk():
    big = seg(0, "y" * 500)
    chunks = chunk_segments([seg(0, "a"), big, seg(2, "b")], max_chars=100)
    assert [len(c) for c in chunks] == [1, 1, 1]
    assert chunks[1] == [big]


def test_empty_input_gives_no_chunks():
    assert chunk_segments([]) == []


def test_non_positive_budget_is_rejected():
    with pytest.raises(ValueError):
        chunk_segments([seg(0, "a")], max_chars=0)


def test_prompt_text_has_timestamps():
    assert to_prompt_text([seg(65, "hello")]) == "[01:05] hello"


def test_paragraphs_split_on_time_window():
    segments = [seg(0, "one"), seg(10, "two"), seg(35, "three"), seg(50, "four")]
    assert paragraphs(segments, window_seconds=30) == [(0, "one two"), (35, "three four")]


def test_transcript_markdown_layout():
    md = transcript_markdown([seg(0, "hi"), seg(40, "bye")], title="Talk", source="https://example.com/v")
    assert md.startswith("# Talk\n\nSource: https://example.com/v\n")
    assert "**[00:00]** hi" in md
    assert "**[00:40]** bye" in md
    assert md.endswith("\n") and not md.endswith("\n\n")
