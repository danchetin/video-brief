from pathlib import Path

import pytest

from video_brief import cli, media
from video_brief.subtitles import Segment


def test_whisper_command_without_prompt():
    cmd = media.whisper_command("whisper-cli", Path("m.bin"), Path("a.wav"), Path("out/w"), lang="ru")
    assert cmd == ["whisper-cli", "-m", "m.bin", "-f", "a.wav", "-l", "ru", "-osrt", "-of", "out/w"]


def test_whisper_command_with_vocabulary_prompt():
    cmd = media.whisper_command("w", Path("m"), Path("a"), Path("o"), lang="en", prompt="Claude, Anthropic")
    assert cmd[-2:] == ["--prompt", "Claude, Anthropic"]


def test_wav_command_is_16khz_mono():
    cmd = media.to_wav_command(Path("in.webm"), Path("out.wav"))
    assert cmd[cmd.index("-ar") + 1] == "16000"
    assert cmd[cmd.index("-ac") + 1] == "1"


def test_captions_command_prefers_original_language_track():
    cmd = media.captions_command("https://x/v", "en", "dir/captions")
    assert cmd[cmd.index("--sub-langs") + 1] == "en-orig,en"
    assert "--skip-download" in cmd


def test_playlists_are_never_downloaded():
    assert "--no-playlist" in media.audio_command("https://x/v", "o")
    assert "--no-playlist" in media.info_command("https://x/v")


def test_transcribe_reports_missing_model(tmp_path):
    with pytest.raises(media.ToolError, match="model not found"):
        media.transcribe(tmp_path / "a.wav", tmp_path / "missing.bin", tmp_path, lang="en")


def test_slugify_and_is_url():
    assert cli.slugify("D-SVFneJ_Nk") == "D-SVFneJ_Nk"
    assert cli.slugify("my talk: part 1/2") == "my-talk-part-1-2"
    assert cli.slugify("///") == "video"
    assert cli.is_url("https://youtu.be/x") and not cli.is_url("talk.mp4")


def test_summarize_requires_api_key(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli.main(["https://youtu.be/x", "--summarize"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_main_writes_transcript_and_brief(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(
        cli, "get_segments", lambda args, workdir: ("abc123", "A Talk", [Segment(0, 1, "hello"), Segment(40, 41, "bye")])
    )
    monkeypatch.setattr(cli, "summarize", lambda segments, **kw: "## Summary\nok")

    code = cli.main(["https://youtu.be/abc123", "--summarize", "--out", str(tmp_path)])

    assert code == 0
    transcript = (tmp_path / "abc123" / "transcript.md").read_text()
    assert "# A Talk" in transcript and "**[00:40]** bye" in transcript
    assert (tmp_path / "abc123" / "brief.md").read_text() == "# A Talk\n\n## Summary\nok\n"


def test_main_reports_tool_errors(monkeypatch, tmp_path, capsys):
    def fail(args, workdir):
        raise media.ToolError("yt-dlp not found")

    monkeypatch.setattr(cli, "get_segments", fail)
    assert cli.main(["https://youtu.be/x", "--out", str(tmp_path)]) == 1
    assert "yt-dlp not found" in capsys.readouterr().err


def test_missing_local_file_is_an_error(capsys):
    assert cli.main(["definitely-not-here.mp4", "--model", "m.bin"]) == 1
    assert "file not found" in capsys.readouterr().err
