"""Thin wrappers around yt-dlp, ffmpeg and whisper.cpp.

Command construction is separated from execution so it can be unit-tested
without the tools installed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .subtitles import Segment, parse_subtitles

WHISPER_BINARIES = ("whisper-cli", "whisper-cpp")


class ToolError(RuntimeError):
    """A required external tool is missing or exited with an error."""


def require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise ToolError(f"`{tool}` not found in PATH (macOS: `brew install {tool}`)")
    return path


def find_whisper() -> str:
    for name in WHISPER_BINARIES:
        path = shutil.which(name)
        if path:
            return path
    raise ToolError("whisper.cpp not found: install it (macOS: `brew install whisper-cpp`)")


def info_command(url: str) -> list[str]:
    return ["yt-dlp", "--no-warnings", "--no-playlist", "--print", "%(id)s\t%(title)s", url]


def audio_command(url: str, out_template: str) -> list[str]:
    return ["yt-dlp", "--no-warnings", "--no-playlist", "-f", "bestaudio", "-o", out_template, url]


def captions_command(url: str, lang: str, out_template: str) -> list[str]:
    return [
        "yt-dlp", "--no-warnings", "--no-playlist", "--skip-download",
        "--write-subs", "--write-auto-subs",
        "--sub-langs", f"{lang}-orig,{lang}", "--sub-format", "vtt",
        "-o", out_template, url,
    ]  # fmt: skip


def to_wav_command(source: Path, target: Path) -> list[str]:
    """16 kHz mono PCM, the format whisper.cpp expects."""
    return [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(target),
    ]  # fmt: skip


def whisper_command(
    binary: str,
    model: Path,
    wav: Path,
    out_base: Path,
    *,
    lang: str,
    prompt: str | None = None,
) -> list[str]:
    command = [
        binary, "-m", str(model), "-f", str(wav), "-l", lang,
        "-osrt", "-of", str(out_base),
    ]  # fmt: skip
    if prompt:
        command += ["--prompt", prompt]
    return command


def _run(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ToolError(f"`{command[0]}` not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or exc.stdout or "").strip().splitlines()[-5:]
        raise ToolError(f"`{command[0]}` failed (exit {exc.returncode}):\n" + "\n".join(tail)) from exc
    return result.stdout


def video_info(url: str) -> tuple[str, str]:
    """Return ``(video_id, title)`` for a URL."""
    require("yt-dlp")
    line = _run(info_command(url)).strip().splitlines()[0]
    video_id, _, title = line.partition("\t")
    return video_id, title or video_id


def fetch_captions(url: str, lang: str, workdir: Path) -> list[Segment] | None:
    """Download existing captions. Returns ``None`` when the video has none."""
    require("yt-dlp")
    _run(captions_command(url, lang, str(workdir / "captions")))
    candidates = sorted(workdir.glob("captions*.vtt"), key=lambda p: ("-orig" not in p.name, p.name))
    if not candidates:
        return None
    return parse_subtitles(candidates[0].read_text(encoding="utf-8"), dedupe_rolling=True)


def fetch_audio(url: str, workdir: Path) -> Path:
    """Download the audio track and convert it to 16 kHz mono WAV."""
    require("yt-dlp")
    require("ffmpeg")
    _run(audio_command(url, str(workdir / "audio.%(ext)s")))
    raw = next(p for p in workdir.glob("audio.*") if p.suffix != ".wav")
    return convert_to_wav(raw, workdir / "audio.wav")


def convert_to_wav(source: Path, target: Path) -> Path:
    require("ffmpeg")
    _run(to_wav_command(source, target))
    return target


def transcribe(wav: Path, model: Path, workdir: Path, *, lang: str, prompt: str | None = None) -> list[Segment]:
    """Run whisper.cpp and parse the SRT it writes."""
    if not model.is_file():
        raise ToolError(
            f"whisper model not found: {model}\n"
            "Download one, e.g. ggml-large-v3-turbo-q5_0.bin from "
            "https://huggingface.co/ggerganov/whisper.cpp"
        )
    out_base = workdir / "whisper"
    _run(whisper_command(find_whisper(), model, wav, out_base, lang=lang, prompt=prompt))
    return parse_subtitles(out_base.with_suffix(".srt").read_text(encoding="utf-8"))
