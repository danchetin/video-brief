# video-brief

[![CI](https://github.com/danchetin/video-brief/actions/workflows/ci.yml/badge.svg)](https://github.com/danchetin/video-brief/actions/workflows/ci.yml)

Turn a long video or podcast into a **timestamped transcript** and, optionally, a **structured brief**
(summary, key ideas, facts, takeaways, caveats). Runs locally: `yt-dlp` fetches the audio,
`whisper.cpp` transcribes it, and an LLM condenses the result if you ask for it.

```text
URL / file ──► yt-dlp ──► ffmpeg (16 kHz mono) ──► whisper.cpp ──► transcript.md
                 │                                                      │
                 └── existing captions (--captions, deduplicated) ──────┤
                                                                        ▼
                                                      chunk ─► notes ─► brief.md   (--summarize)
```

## Why this exists

I wanted to pull facts and ideas out of 45-minute interviews without watching them in full.
Pasting a transcript into a chat window breaks down for long videos, so this tool handles the
parts that are annoying to do by hand: getting clean timestamped text, fitting it into an LLM
context, and keeping the output traceable to the moment in the video.

## Install

Requires Python 3.10+ and three external tools:

```bash
brew install yt-dlp ffmpeg whisper-cpp        # macOS; use your package manager elsewhere
pip install -e .                              # no Python dependencies
```

Download a whisper.cpp model once (about 550 MB, good speed/quality balance):

```bash
curl -L -o ggml-large-v3-turbo-q5_0.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin
export VIDEO_BRIEF_MODEL=$PWD/ggml-large-v3-turbo-q5_0.bin
```

## Usage

```bash
# transcript only
video-brief "https://www.youtube.com/watch?v=VIDEO_ID"

# transcript + brief (needs ANTHROPIC_API_KEY)
video-brief "https://www.youtube.com/watch?v=VIDEO_ID" --summarize

# help whisper spell names it would otherwise guess wrong
video-brief VIDEO_URL --vocab "Claude, Anthropic" --summarize

# Russian audio, or a local file
video-brief talk.mp4 --lang ru

# skip transcription and reuse the video's own captions (fast, lower quality)
video-brief VIDEO_URL --captions
```

Output goes to `out/<video-id>/`:

```text
transcript.md    # paragraphs with a [MM:SS] marker, roughly every 30 seconds
brief.md         # only with --summarize
```

Example `brief.md` structure:

```markdown
# Talk title

## Summary
...
## Key ideas
- [13:03] ...
## Facts and curiosities
## Practical takeaways
## Caveats
```

## Design notes

- **Standard library only.** The Anthropic call is plain `urllib`; the package has no runtime
  dependencies to install or audit. External tools are checked up front with install hints.
- **Map-reduce summarization.** Transcripts are split on segment boundaries into chunks under a
  character budget, each chunk becomes timestamped notes, and the notes are merged into one brief.
  Timestamps survive both steps, so every point in the brief can be found in the video.
- **Transcript text is treated as untrusted data.** It comes from the internet, so the prompts
  tell the model not to follow instructions that appear inside it.
- **YouTube auto-captions are deduplicated.** Their WebVTT repeats the previous line in every cue.
  On a 49-minute video the parser reduces 3378 cues to 1690 without losing text.
- **Vocabulary hints.** `--vocab` passes an initial prompt to whisper. Without it, a speaker
  saying "Claude" was transcribed as "cloud" in places; with it, the spelling is correct.
- **Command building is separate from execution**, so the exact `yt-dlp` / `ffmpeg` /
  `whisper-cli` invocations are unit-tested without the tools installed.
- **No keys in output.** API errors are reported with status and a short body excerpt, never the key.

## Performance

Measured on an Apple Silicon Mac with `ggml-large-v3-turbo-q5_0`: a 49-minute talk transcribes
in about 4.7 minutes (roughly 10x real time).

## Limitations

- **No speaker diarization.** whisper.cpp returns one stream of text without "who said it".
  The brief tells the model to name speakers only when the context makes it clear.
- **Speech recognition makes mistakes**, especially with names and jargon. The brief has a
  *Caveats* section for this, but check anything you plan to quote.
- The summarization step is covered by tests with a fake HTTP layer. Quality depends on the
  model you choose with `--llm-model`.
- Downloading videos is subject to the platform's terms; use it for content you may process.

## Development

```bash
pip install -e ".[dev]"
pytest
```

CI runs the tests on Python 3.10, 3.12 and 3.13.

## License

MIT
