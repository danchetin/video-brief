import json
import urllib.error

import pytest

from video_brief import summarize as module
from video_brief.subtitles import Segment
from video_brief.summarize import SummarizeError, summarize


class FakePost:
    """Records requests and returns canned answers."""

    def __init__(self):
        self.calls = []

    def __call__(self, payload, api_key):
        self.calls.append((payload, api_key))
        prompt = payload["messages"][0]["content"]
        return "FINAL BRIEF" if "<notes>" in prompt else f"notes#{len(self.calls)}"


def segments(n, size=60):
    return [Segment(i * 10, i * 10 + 9, "w" * size) for i in range(n)]


def test_map_reduce_calls_once_per_chunk_plus_final():
    post = FakePost()
    result = summarize(segments(10), api_key="k", max_chars=200, post=post)
    chunks = len(post.calls) - 1
    assert chunks >= 2
    assert result == "FINAL BRIEF"
    final_prompt = post.calls[-1][0]["messages"][0]["content"]
    assert "### Part 1" in final_prompt and f"### Part {chunks}" in final_prompt


def test_single_chunk_skips_part_headings():
    post = FakePost()
    summarize(segments(1), api_key="k", post=post)
    assert len(post.calls) == 2
    assert "### Part" not in post.calls[-1][0]["messages"][0]["content"]


def test_model_and_key_are_forwarded():
    post = FakePost()
    summarize(segments(1), api_key="secret", model="some-model", post=post)
    payload, key = post.calls[0]
    assert payload["model"] == "some-model" and key == "secret"


def test_transcript_is_marked_as_untrusted_data():
    post = FakePost()
    summarize(segments(1), api_key="k", post=post)
    prompt = post.calls[0][0]["messages"][0]["content"]
    assert "untrusted data" in prompt and "<transcript>" in prompt


def test_empty_transcript_is_an_error():
    with pytest.raises(SummarizeError):
        summarize([], api_key="k", post=FakePost())


def test_http_errors_do_not_leak_the_key(monkeypatch):
    def boom(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, _Body(b'{"error":"bad key"}'))

    monkeypatch.setattr(module.urllib.request, "urlopen", boom)
    with pytest.raises(SummarizeError) as info:
        module._post({"model": "m"}, "sk-very-secret")
    assert "401" in str(info.value)
    assert "sk-very-secret" not in str(info.value)


def test_post_joins_text_blocks(monkeypatch):
    class Response(_Body):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    body = json.dumps({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}).encode()
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda request, timeout: Response(body))
    assert module._post({"model": "m"}, "k") == "ab"


class _Body:
    def __init__(self, data):
        import io

        self._buffer = io.BytesIO(data)

    def read(self, *args):
        return self._buffer.read(*args)

    def close(self):
        pass
