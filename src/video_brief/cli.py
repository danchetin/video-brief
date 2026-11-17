"""Command line interface: ``video-brief <url-or-file> [options]``."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

from . import __version__, media
from .render import transcript_markdown
from .subtitles import Segment
from .summarize import DEFAULT_MODEL, SummarizeError, summarize

MODEL_ENV = "VIDEO_BRIEF_MODEL"
KEY_ENV = "ANTHROPIC_API_KEY"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-brief",
        description="Turn a video (URL or local file) into a timestamped transcript and an optional brief.",
    )
    parser.add_argument("source", help="video URL (anything yt-dlp supports) or a local audio/video file")
    parser.add_argument("-l", "--lang", default="en", help="spoken language code (default: en)")
    parser.add_argument("-o", "--out", type=Path, default=Path("out"), help="output directory (default: ./out)")
    parser.add_argument(
        "-m", "--model", type=Path,
        help=f"whisper.cpp model file (or set {MODEL_ENV})",
    )  # fmt: skip
    parser.add_argument(
        "--captions", action="store_true",
        help="use the video's existing captions instead of running whisper (faster, lower quality)",
    )  # fmt: skip
    parser.add_argument(
        "--vocab", metavar="TERMS",
        help='comma-separated names/terms to help whisper spell them, e.g. "Claude, Anthropic"',
    )  # fmt: skip
    parser.add_argument("--summarize", action="store_true", help=f"also write brief.md using the Anthropic API ({KEY_ENV})")
    parser.add_argument("--llm-model", default=DEFAULT_MODEL, help=f"model for --summarize (default: {DEFAULT_MODEL})")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return slug or "video"


def is_url(source: str) -> bool:
    return re.match(r"^https?://", source) is not None


def log(message: str) -> None:
    print(message, file=sys.stderr)


def get_segments(args: argparse.Namespace, workdir: Path) -> tuple[str, str, list[Segment]]:
    """Return ``(video_id, title, segments)`` for the requested source."""
    if is_url(args.source):
        video_id, title = media.video_info(args.source)
        if args.captions:
            log("Downloading captions...")
            segments = media.fetch_captions(args.source, args.lang, workdir)
            if segments is None:
                raise media.ToolError("this video has no captions; run without --captions")
            return video_id, title, segments
        log("Downloading audio...")
        wav = media.fetch_audio(args.source, workdir)
    else:
        path = Path(args.source)
        if not path.is_file():
            raise media.ToolError(f"file not found: {path}")
        if args.captions:
            raise media.ToolError("--captions only works with URLs")
        video_id, title = slugify(path.stem), path.stem
        log("Converting audio...")
        wav = media.convert_to_wav(path, workdir / "audio.wav")

    model_path = args.model or os.environ.get(MODEL_ENV)
    if not model_path:
        raise media.ToolError(f"no whisper model: pass --model or set {MODEL_ENV}")
    log("Transcribing with whisper.cpp (this can take a few minutes)...")
    segments = media.transcribe(wav, Path(model_path), workdir, lang=args.lang, prompt=args.vocab)
    return video_id, title, segments


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    api_key = os.environ.get(KEY_ENV, "")
    if args.summarize and not api_key:
        log(f"error: --summarize needs {KEY_ENV} in the environment")
        return 2

    try:
        with tempfile.TemporaryDirectory(prefix="video-brief-") as tmp:
            video_id, title, segments = get_segments(args, Path(tmp))

        if not segments:
            log("error: the transcript is empty")
            return 1

        out_dir = args.out / slugify(video_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        source = args.source if is_url(args.source) else Path(args.source).name
        transcript_path = out_dir / "transcript.md"
        transcript_path.write_text(transcript_markdown(segments, title=title, source=source), encoding="utf-8")
        log(f"Wrote {transcript_path}")

        if args.summarize:
            log("Summarizing...")
            brief = summarize(segments, api_key=api_key, model=args.llm_model)
            brief_path = out_dir / "brief.md"
            brief_path.write_text(f"# {title}\n\n{brief}\n", encoding="utf-8")
            log(f"Wrote {brief_path}")
    except (media.ToolError, SummarizeError) as exc:
        log(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
