# transcribe

Local on-Mac audio → timestamped transcript pipeline. Accepts Olympus
`.ds2`/`.dss` recordings (via the sibling `ds2-converter` project) as well as
`.mp3`/`.wav`/`.m4a`/`.flac`/`.opus`/`.ogg`/`.aac`, normalizes everything to
16 kHz mono WAV, runs Silero VAD to skip silence, then runs
[mlx-whisper] (`whisper-large-v3-turbo` on Apple MLX). Output is a
`.transcript.txt` (segment-level timestamps inline) and a `.transcript.json`
sidecar (full Whisper output incl. word timestamps).

The transcript is meant to be pasted into a Claude.ai project for the final
stitch / cleanup step — that step is intentionally *not* automated here.

## Layout

```
~/Audio/Inbox/                       you drop files here
~/Audio/Done/YYYY-MM-DD/             sources move here after success
~/Audio/Out/YYYY-MM-DD/              transcripts land here
~/Audio/Prompts/default.txt          (optional) glossary loaded automatically
```

## Prerequisites

```bash
brew install ffmpeg pipx
pipx ensurepath
# ds2-converter must already be `npm link`-ed so `ds2-convert` is on PATH
```

## Install

```bash
cd ~/Code/transcribe
pipx install --editable .
pipx inject transcribe silero-vad     # for VAD pre-segmentation
```

## Usage

```bash
# Process everything in ~/Audio/Inbox/
transcribe

# Or transcribe specific files (sources are left in place)
transcribe path/to/recording1.ds2 path/to/recording2.mp3
```

The first run downloads `whisper-large-v3-turbo` weights (~800 MB) into
`~/.cache/huggingface/hub/`. Subsequent runs are cached.

## What's in the pipeline (in order)

1. **Format detection** — `.ds2`/`.dss` go through `ds2-convert`; everything
   else goes through `ffmpeg` for normalization to 16 kHz mono WAV.
2. **Silero VAD** — speech regions are detected and merged aggressively (gaps
   < 2 s collapsed, blips < 0.5 s dropped, ±0.5 s padding). Silence between
   speech regions is skipped from Whisper entirely. Disable with `--no-vad`.
3. **mlx-whisper** transcription with `whisper-large-v3-turbo` (English
   accuracy parity with `large-v3` at 6-8× speed). Decoder tuned for
   solo dictation: `condition_on_previous_text=False`,
   `hallucination_silence_threshold=2.0`, plus tightened
   `no_speech_threshold`, `compression_ratio_threshold`, `logprob_threshold`.
4. **Initial prompt** — a 224-token glossary/style hint that biases the
   decoder's vocabulary priors. See "Prompts" below.
5. **Bag-of-Hallucinations filter** — drops segments matching known Whisper
   hallucinations ("Thank you for watching", "you", "Bye", etc.) and
   segments with within-segment 3-gram looping. Per Barański et al.,
   arxiv 2501.11378.

## Prompts

A 224-token (~150-180 word) prefix is sent to Whisper to bias vocabulary
and style priors. Three sources are checked, highest priority first:

1. `--prompt "..."` or `--prompt-file PATH` on the command line.
2. Sidecar file `<audio>.prompt` in the same directory as the audio
   (e.g., `~/Audio/Inbox/foo.mp3.prompt`).
3. `~/Audio/Prompts/default.txt` (loaded automatically if it exists).

`--no-prompt` disables all sources for one run.

**What works in a prompt:** glossary of proper nouns spelled the way you
want them, domain vocabulary, 1-2 sentences in your preferred style. The
model follows the *style* of the prompt — it does NOT follow instructions
like "transcribe carefully" or "fix grammar."

**What doesn't work:** instructions, long backstory (anything past 224
tokens is silently dropped), style hints that contradict the audio.

**Known limitation:** because `condition_on_previous_text=False` (an
anti-hallucination must), the prompt only seeds the **first ~30 s** of
each recording. Names mentioned later in a long file decode fresh and
won't get the prompt's vocabulary bias. The Whisper architecture doesn't
have a clean way around this.

## Output format

`recording.transcript.txt`:

```
[00:00.000 → 00:03.420]  Hello, this is a test recording.
[00:03.420 → 00:07.812]  Thinking out loud about a few things today.
```

`recording.transcript.json` contains the full mlx-whisper result dict
(segments with word-level timestamps, language detection, etc.).

## CLI flags

```
transcribe [--quiet | --show-segments]
           [--prompt TEXT | --prompt-file PATH | --no-prompt]
           [--no-vad]
           [--model MODEL]
           [files ...]
```

| Flag | Effect |
|------|--------|
| (default) | tqdm progress bar (% through audio + ETA) |
| `-v` / `--show-segments` | streams each Whisper segment to stdout as decoded |
| `-q` / `--quiet` | silent until each file finishes |
| `--prompt TEXT` | inline initial_prompt (≤224 tokens; overrides sidecar/default) |
| `--prompt-file PATH` | read prompt from a file (overrides sidecar/default) |
| `--no-prompt` | disable all prompt sources for this run |
| `--no-vad` | skip Silero VAD pre-segmentation (transcribe whole audio) |
| `--model MODEL` | override the HuggingFace MLX model id |

## Performance notes (M1 Pro 16 GB)

| Config | RTF (12 min file) | Notes |
|--------|--------------------|-------|
| large-v3 + decoder defaults | 0.22x | original baseline |
| large-v3-turbo + tightened decoder | 0.12x | speed win, English parity |
| + initial prompt | 0.12x | no speed cost |
| + BoH filter | 0.12x | trivial post-process |
| + Silero VAD (default) | 0.20x | ~1.7x slower, restores some detail lost to tightened thresholds, skips silence |

The VAD slowdown is real because each VAD region is a separate Whisper
seek. We use aggressive merging (`min_silence_duration_ms=2000`) so a
typical recording yields ~30-60 regions, not hundreds. If you don't need
VAD's anti-hallucination defense for a clean recording, `--no-vad`
restores the faster path.

## Known footguns

- **`word_timestamps=True` memory leak** in long batch jobs
  (mlx-examples #1254). Single-file invocations are unaffected; a future
  launchd watcher should restart the process between batches.
- **Prompt-leakage**: Whisper occasionally inserts prompt vocabulary into
  the transcript during silent stretches. Mitigated by VAD + BoH filter,
  but worth eyeballing if a recording has long silences.
- **dss-codec stuck-output regions**: the upstream WASM decoder
  (`hirparak/dss-codec`) occasionally loses sync on certain DS2 byte
  patterns and emits a constant sample value (typically ±32767) for
  several seconds before recovering. Whisper hears the resulting flat
  tone as "voice-like" and hallucinates gibberish on top of it. The
  pipeline detects these regions and prints a warning like:

  ```
  ⚠ dss-codec stuck: 1 constant-value region(s), 11.9s lost
      18.7s →   30.5s  (constant sample value -32767)
  ```

  If you see this warning, the corresponding seconds of audio are
  unrecoverable and the transcript around them is likely hallucination.
  The detector only flags the *symptom*; the *cause* lives upstream in
  the codec.

[mlx-whisper]: https://github.com/ml-explore/mlx-examples/tree/main/whisper
