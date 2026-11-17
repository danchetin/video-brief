import pytest

from video_brief.subtitles import Segment, format_timestamp, parse_subtitles, parse_timestamp

SRT = """1
00:00:00,000 --> 00:00:02,500
 Hello there.

2
00:00:02,500 --> 00:00:05,000
 This is the second
 cue on two lines.
"""

# Shaped like YouTube auto-captions: every cue repeats the previous cue's last line,
# and tiny 10 ms cues only carry the repeated line.
ROLLING_VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:01.390 align:start position:0%

If<00:00:00.240><c> I</c><00:00:00.360><c> were</c><00:00:00.480><c> doing</c><00:00:00.720><c> this</c>

00:00:01.390 --> 00:00:01.400 align:start position:0%
If I were doing this


00:00:01.400 --> 00:00:02.550 align:start position:0%
If I were doing this
summarize<00:00:01.560><c> this</c><00:00:01.800><c> email</c>

00:00:02.550 --> 00:00:02.560 align:start position:0%
summarize this email


00:00:02.560 --> 00:00:04.000 align:start position:0%
summarize this email
you<00:00:02.600><c> are</c><00:00:02.700><c> not</c><00:00:02.800><c> benefiting</c>
"""


def test_parse_timestamp_variants():
    assert parse_timestamp("00:01:02,500") == 62.5
    assert parse_timestamp("01:02.250") == 62.25
    assert parse_timestamp("1:00:00.000") == 3600


def test_parse_timestamp_rejects_garbage():
    with pytest.raises(ValueError):
        parse_timestamp("soon")


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "00:00"), (59.9, "00:59"), (61, "01:01"), (3600, "1:00:00"), (3725, "1:02:05"), (-3, "00:00")],
)
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_parse_srt_multiline_cues():
    assert parse_subtitles(SRT) == [
        Segment(0.0, 2.5, "Hello there."),
        Segment(2.5, 5.0, "This is the second cue on two lines."),
    ]


def test_parse_srt_handles_windows_newlines():
    assert len(parse_subtitles(SRT.replace("\n", "\r\n"))) == 2


def test_blocks_without_timing_are_ignored():
    assert parse_subtitles("WEBVTT\n\nNOTE just a comment\n\n42\n") == []


def test_tags_are_stripped_without_dedupe():
    segments = parse_subtitles(ROLLING_VTT)
    assert segments[0].text == "If I were doing this"


def test_rolling_captions_are_deduplicated():
    segments = parse_subtitles(ROLLING_VTT, dedupe_rolling=True)
    assert [s.text for s in segments] == [
        "If I were doing this",
        "summarize this email",
        "you are not benefiting",
    ]
