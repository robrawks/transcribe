"""transcribe — local audio → timestamped transcript pipeline.

Default mode: scan ~/Audio/Inbox/ and transcribe every supported file.
Files passed explicitly are transcribed in place (source not moved).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from collections import Counter

import mlx_whisper

INBOX = Path.home() / "Audio" / "Inbox"
DONE = Path.home() / "Audio" / "Done"
OUT = Path.home() / "Audio" / "Out"
PROMPTS_DIR = Path.home() / "Audio" / "Prompts"
DEFAULT_PROMPT_FILE = PROMPTS_DIR / "default.txt"

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

DS2_EXTS = {".ds2", ".dss"}
OTHER_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".opus", ".ogg", ".aac", ".aiff", ".aif"}

# Documented Whisper "bag of hallucinations" — short phrases the model
# emits on silent or non-speech segments due to YouTube-heavy training data.
# Per Barański et al., arxiv 2501.11378 (Jan 2025).
BOH_PHRASES = frozenset([
    "you",
    "thank you",
    "thank you.",
    "thank you!",
    "thank you for watching",
    "thank you for watching.",
    "thank you for watching!",
    "thanks for watching",
    "thanks for watching.",
    "thanks for watching!",
    "bye",
    "bye.",
    "bye!",
    "goodbye",
    "goodbye.",
    "subscribe",
    "subscribe.",
    "please subscribe",
    "please subscribe.",
    "like and subscribe",
    "like and subscribe.",
    "please like and subscribe",
    "please like and subscribe.",
])


def format_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:06.3f}"


def detect_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in DS2_EXTS:
        return "ds2"
    if ext in OTHER_AUDIO_EXTS:
        return "other_audio"
    return "unsupported"


def convert_ds2(src: Path, tmpdir: Path) -> Path:
    subprocess.run(
        ["ds2-convert", "--out-dir", str(tmpdir), "--quiet", str(src)],
        check=True,
    )
    wav = tmpdir / (src.stem + ".wav")
    if not wav.is_file():
        raise FileNotFoundError(f"ds2-convert did not produce {wav}")
    return wav


# DS2 file structure constants and expected compressed bitrate. The header is
# always 0x600 bytes; everything after that is compressed audio. For QP mode
# (the only mode the DS-5000 records in by default), observed payload bitrate
# across ~20 healthy files is tightly clustered at 3543-3556 bytes/sec.
DS2_HEADER_BYTES = 0x600
DS2_QP_EXPECTED_BYTES_PER_SEC = 3545
# Trigger only on significant overshoot (≥20% above expected). The actual
# observed anomaly on DS500339 was 5131 bytes/sec (45% above). A 20% floor
# leaves comfortable headroom against the 2.8s-file outlier (3721, +5%).
DS2_BITRATE_ANOMALY_RATIO = 1.20


def check_ds2_bitrate(src: Path, wav: Path) -> tuple[float, float] | None:
    """Compare the DS2 file's compressed bitrate against the expected ~3545
    bytes/sec for QP mode. Returns (observed_bps, expected_bps) if the file
    has consumed significantly more input bytes per output second than a
    healthy QP recording would; returns None otherwise.

    This is a second, complementary detector for the same upstream
    hirparak/dss-codec failure mode that detect_constant_runs() catches.
    When the WASM decoder loses sync, it consumes extra input bytes while
    producing fewer output samples. The bitrate anomaly is sometimes the
    only visible symptom — the codec may stay just below the constant-run
    detection threshold yet still mis-decode chunks of the file.

    The DS-5000 records exclusively in QP mode (the SP/LP modes exist in
    spec but require explicit menu changes that this device's owner has
    never made). If the recorder is ever reconfigured to SP/LP, the
    observed bitrate would legitimately drop — only over-shoot triggers
    the warning, so a lower-rate mode would never false-positive here.
    """
    import wave

    file_size = src.stat().st_size
    payload_bytes = file_size - DS2_HEADER_BYTES
    if payload_bytes <= 0:
        return None

    with wave.open(str(wav), "rb") as wf:
        sps = wf.getframerate()
        frames = wf.getnframes()

    if sps == 0 or frames == 0:
        return None
    duration_sec = frames / sps

    observed = payload_bytes / duration_sec
    threshold = DS2_QP_EXPECTED_BYTES_PER_SEC * DS2_BITRATE_ANOMALY_RATIO
    if observed > threshold:
        return (observed, float(DS2_QP_EXPECTED_BYTES_PER_SEC))
    return None


def convert_with_ffmpeg(src: Path, tmpdir: Path) -> Path:
    wav = tmpdir / (src.stem + ".wav")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src),
            "-ac", "1",
            "-ar", "16000",
            str(wav),
        ],
        check=True,
    )
    return wav


def detect_constant_runs(
    wav_path: Path,
    min_seconds: float = 1.0,
) -> list[tuple[float, float, int]]:
    """Find contiguous runs of identical int16 samples ≥ `min_seconds` long.

    The hirparak/dss-codec WASM decoder (the engine inside ds2-convert) has a
    known failure mode where it loses sync on certain DS2 byte patterns and
    emits a constant sample value (typically ±32767) for seconds at a time
    before either recovering or staying stuck for the rest of the file. The
    user hears a flat tone; Whisper hears the loud "voice-like" steady-state
    and hallucinates gibberish (Hebrew Unicode, "Morning Morning Morning...",
    etc.) on top of legitimate speech in the rest of the recording.

    This detector flags those regions so we can warn the user rather than
    silently produce a confusing transcript. Returns a list of
    (start_sec, end_sec, sample_value) tuples; empty list if the WAV is fine.
    """
    import wave
    import numpy as np

    with wave.open(str(wav_path), "rb") as wf:
        sps = wf.getframerate()
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    if len(samples) == 0 or sps == 0:
        return []

    min_run = max(int(min_seconds * sps), 1)
    # Find the boundaries where the sample value changes.
    diffs = np.diff(samples)
    change_idx = np.nonzero(diffs)[0] + 1  # indices where a new run starts
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(samples)]))
    runs: list[tuple[float, float, int]] = []
    for s, e in zip(starts, ends):
        if e - s >= min_run:
            runs.append((s / sps, e / sps, int(samples[s])))
    return runs


def _load_pcm_for_vad(wav_path: Path) -> "object":
    """Load `wav_path` as a 16 kHz mono float32 torch tensor, via ffmpeg.

    Bypasses silero-vad's `read_audio`, which depends on a torchaudio version
    that requires torchcodec on modern installs. We already have ffmpeg on
    PATH, so this is the lighter dependency surface.
    """
    import numpy as np
    import torch

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(wav_path),
        "-f", "f32le", "-ac", "1", "-ar", "16000",
        "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    arr = np.frombuffer(out, dtype=np.float32)
    return torch.from_numpy(arr.copy())  # copy to make the buffer writable


def vad_clip_timestamps(wav_path: Path) -> str | None:
    """Run Silero VAD on `wav_path`, return Whisper-style clip_timestamps string.

    Format: "s1,e1,s2,e2,..." in seconds. Whisper internally maps these back to
    absolute audio time, so the resulting segments need no offset adjustment.
    Returns None if no speech was detected (caller falls back to whole-file).

    Silero VAD is 1.8 MB, runs in ~1 ms per 30 ms chunk. Per Calm-Whisper paper
    (arxiv 2505.12969), pre-segmentation eliminates 80%+ of non-speech
    hallucinations with <0.1% WER cost.
    """
    from silero_vad import load_silero_vad, get_speech_timestamps

    model = load_silero_vad()
    wav = _load_pcm_for_vad(wav_path)
    # Aggressive merging on purpose: Whisper has its own 30s decoder window,
    # so feeding it many tiny clips wrecks throughput AND adds edge-effect
    # hallucinations. We want a handful of clips averaging ≥30 s each.
    stamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=16000,
        return_seconds=True,
        min_silence_duration_ms=2000,  # merge speech across silences up to 2 s
        min_speech_duration_ms=500,    # drop blips shorter than 0.5 s
        speech_pad_ms=500,             # ±0.5 s padding to preserve word edges
    )
    if not stamps:
        return None
    parts: list[str] = []
    for s in stamps:
        parts.append(f"{s['start']:.2f}")
        parts.append(f"{s['end']:.2f}")
    return ",".join(parts)


def _has_ngram_loop(text: str, n: int = 3, threshold: int = 3) -> bool:
    """Detect within-segment repetition. True if any n-gram appears ≥threshold times."""
    words = text.split()
    if len(words) < n * threshold:
        return False
    ngrams = Counter(
        " ".join(words[i:i + n]).lower() for i in range(len(words) - n + 1)
    )
    return any(c >= threshold for c in ngrams.values())


def filter_hallucinations(result: dict) -> int:
    """Drop empty / blocklisted / looping segments from `result` in place.
    Returns the number of segments dropped.
    """
    segs = result.get("segments") or []
    kept = []
    dropped = 0
    for seg in segs:
        text = (seg.get("text") or "").strip()
        if not text:
            dropped += 1
            continue
        if text.lower() in BOH_PHRASES:
            dropped += 1
            continue
        if _has_ngram_loop(text):
            dropped += 1
            continue
        kept.append(seg)
    result["segments"] = kept
    return dropped


def _norm_words(s: str) -> list[str]:
    """Lowercase + strip outer punctuation. For prompt-echo matching."""
    return [w.lower().strip(",.!?:;\"'-—…") for w in s.split() if w.strip()]


def strip_prompt_echo(result: dict, initial_prompt: str | None) -> int:
    """Detect and strip prompt-tail echoed into the start of the transcript.

    Whisper occasionally emits the trailing portion of its `initial_prompt`
    into the first audible segment (documented prompt-leakage failure mode).
    We find the longest word sequence that is BOTH a suffix of the prompt
    AND a prefix of the first segment, with case-insensitive +
    punctuation-tolerant matching. Require ≥3 words match so we don't
    false-positive on incidental words like "I'm" or "the".

    If the entire first segment turned out to be prompt echo, the segment
    is dropped. Otherwise we trim the prefix off the segment text.

    Returns the number of words stripped (0 if no echo detected).
    """
    if not initial_prompt:
        return 0
    segs = result.get("segments") or []
    if not segs:
        return 0
    first = segs[0]
    seg_text = (first.get("text") or "").strip()
    if not seg_text:
        return 0
    prompt_n = _norm_words(initial_prompt)
    seg_n = _norm_words(seg_text)
    if not prompt_n or not seg_n:
        return 0
    max_k = min(len(prompt_n), len(seg_n), 25)
    for k in range(max_k, 2, -1):           # require ≥3-word match
        if prompt_n[-k:] == seg_n[:k]:
            seg_words = seg_text.split()
            remaining = seg_words[k:]
            if remaining:
                first["text"] = " " + " ".join(remaining)
                return k
            segs.pop(0)
            return k
    return 0


def _read_prompt_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def resolve_prompt(
    src: Path,
    cli_prompt: str | None,
    cli_prompt_file: Path | None,
    no_prompt: bool,
) -> tuple[str | None, str]:
    """Resolve which initial_prompt (if any) to send to Whisper for `src`.

    Precedence (highest first):
      --no-prompt > --prompt > --prompt-file > sidecar > default file > none.
    Returns (prompt_text_or_None, human_source_label).
    """
    if no_prompt:
        return None, "disabled (--no-prompt)"
    if cli_prompt is not None:
        return cli_prompt.strip(), "--prompt"
    if cli_prompt_file is not None:
        if not cli_prompt_file.is_file():
            return None, f"WARN: --prompt-file {cli_prompt_file} not found"
        return _read_prompt_file(cli_prompt_file), f"--prompt-file {cli_prompt_file}"
    sidecar = src.parent / (src.name + ".prompt")
    if sidecar.is_file():
        return _read_prompt_file(sidecar), f"sidecar {sidecar.name}"
    if DEFAULT_PROMPT_FILE.is_file():
        return _read_prompt_file(DEFAULT_PROMPT_FILE), f"default ({DEFAULT_PROMPT_FILE})"
    return None, "no prompt source"


def render_transcript_txt(result: dict) -> str:
    lines = []
    for seg in result.get("segments", []):
        start = format_ts(seg.get("start", 0.0))
        end = format_ts(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
        lines.append(f"[{start} → {end}]  {text}")
    return ("\n".join(lines) + "\n") if lines else ""


def audio_duration(result: dict) -> float:
    segs = result.get("segments") or []
    if not segs:
        return 0.0
    return float(segs[-1].get("end", 0.0))


def transcribe_one(
    src: Path,
    today: str,
    model: str,
    from_inbox: bool,
    verbose: bool | None,
    initial_prompt: str | None,
    use_vad: bool,
) -> tuple[bool, str]:
    fmt = detect_format(src)
    if fmt == "unsupported":
        return False, f"unsupported format: {src.name}"

    out_dir = OUT / today
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="transcribe-") as tmp:
        tmpdir = Path(tmp)
        try:
            if fmt == "ds2":
                wav = convert_ds2(src, tmpdir)
            else:
                wav = convert_with_ffmpeg(src, tmpdir)
        except subprocess.CalledProcessError as e:
            return False, f"decode/convert failed for {src.name}: {e}"
        except FileNotFoundError as e:
            return False, f"decode/convert failed for {src.name}: {e}"

        stuck_runs = detect_constant_runs(wav)
        if stuck_runs:
            total_lost = sum(e - s for s, e, _ in stuck_runs)
            print(
                f"  ⚠ dss-codec stuck: {len(stuck_runs)} constant-value "
                f"region(s), {total_lost:.1f}s lost"
            )
            for s, e, v in stuck_runs:
                print(f"      {s:6.1f}s → {e:6.1f}s  (constant sample value {v})")

        if fmt == "ds2":
            bitrate_anomaly = check_ds2_bitrate(src, wav)
            if bitrate_anomaly is not None:
                observed, expected = bitrate_anomaly
                pct = 100 * (observed / expected - 1)
                print(
                    f"  ⚠ ds2 bitrate anomaly: {observed:.0f} bytes/sec "
                    f"(expected ~{expected:.0f} for ds2_qp, +{pct:.0f}%). "
                    f"Decoder likely mis-decoded — preserve {src.name} "
                    f"for re-decoding with another tool."
                )

        clip_ts: str = "0"  # default: transcribe whole file
        vad_label = "VAD disabled"
        if use_vad:
            try:
                vad_result = vad_clip_timestamps(wav)
            except Exception as e:
                vad_result = None
                vad_label = f"VAD failed ({e}); transcribing full file"
            if vad_result is None:
                if "failed" not in vad_label:
                    vad_label = "VAD: no speech detected; transcribing full file"
            else:
                clip_ts = vad_result
                vad_label = f"VAD: {len(clip_ts.split(',')) // 2} speech region(s)"
        print(f"  {vad_label}")

        try:
            result = mlx_whisper.transcribe(
                str(wav),
                path_or_hf_repo=model,
                word_timestamps=True,
                verbose=verbose,
                clip_timestamps=clip_ts,
                # Anti-hallucination defaults tuned for solo dictation with pauses.
                # condition_on_previous_text=True (the library default) causes
                # cascading stuck-loop hallucinations during silence. False makes
                # each 30s window decode independently — far fewer repeats, at
                # a mild cost to cross-window punctuation continuity.
                condition_on_previous_text=False,
                # Actively drop repetitive segments during silence ≥ 2s.
                hallucination_silence_threshold=2.0,
                # Tightened thresholds vs library defaults — modestly more
                # aggressive about catching gibberish/silence on clean speech.
                # Sources: OpenAI Cookbook + openai/whisper discussion #679.
                no_speech_threshold=0.7,            # default 0.6
                compression_ratio_threshold=2.0,    # default 2.4
                logprob_threshold=-0.8,             # default -1.0
                # 224-token glossary/style prompt that biases the decoder's
                # vocabulary priors. See resolve_prompt() for resolution order.
                initial_prompt=initial_prompt,
            )
        except Exception as e:
            return False, f"whisper failed for {src.name}: {e}"

    dropped = filter_hallucinations(result)
    echo_stripped = strip_prompt_echo(result, initial_prompt)

    base = src.stem
    txt_path = out_dir / f"{base}.transcript.txt"
    txt_path.write_text(render_transcript_txt(result), encoding="utf-8")

    if from_inbox:
        done_dir = DONE / today
        done_dir.mkdir(parents=True, exist_ok=True)
        dest = done_dir / src.name
        if dest.exists():
            stamp = dt.datetime.now().strftime("%H%M%S")
            dest = done_dir / f"{src.stem}.{stamp}{src.suffix}"
        shutil.move(str(src), str(dest))

    duration = audio_duration(result)
    wall = time.monotonic() - started
    rtf = (wall / duration) if duration > 0 else float("nan")
    extras = ""
    if dropped:
        extras += f"  filtered={dropped}"
    if echo_stripped:
        extras += f"  echo_stripped={echo_stripped}w"
    return True, (
        f"ok: {src.name}  duration={duration:.1f}s  wall={wall:.1f}s  "
        f"RTF={rtf:.2f}x{extras}"
    )


def gather_inputs(args_files: list[str]) -> tuple[list[Path], bool]:
    if args_files:
        return [Path(f).expanduser().resolve() for f in args_files], False
    if not INBOX.is_dir():
        return [], True
    files = sorted(
        p for p in INBOX.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )
    return files, True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Local audio → timestamped transcript via mlx-whisper.",
    )
    parser.add_argument(
        "files", nargs="*",
        help="Specific files to transcribe. If omitted, scans ~/Audio/Inbox/.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"HuggingFace repo for the Whisper MLX model (default: {DEFAULT_MODEL}).",
    )
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress the progress bar (silent until each file finishes).",
    )
    progress.add_argument(
        "--show-segments", "-v", action="store_true",
        help="Stream each Whisper segment to stdout as it's decoded "
             "(replaces the progress bar).",
    )

    prompt_grp = parser.add_mutually_exclusive_group()
    prompt_grp.add_argument(
        "--prompt", metavar="TEXT",
        help="Inline initial_prompt for Whisper (224-token limit). "
             "Overrides sidecar and default prompt file.",
    )
    prompt_grp.add_argument(
        "--prompt-file", metavar="PATH",
        help="Path to a prompt file. Overrides sidecar and default prompt file.",
    )
    prompt_grp.add_argument(
        "--no-prompt", action="store_true",
        help="Disable all prompt sources for this run "
             f"(ignores {DEFAULT_PROMPT_FILE} and any sidecar files).",
    )
    parser.add_argument(
        "--no-vad", action="store_true",
        help="Disable Silero VAD pre-segmentation (transcribe entire audio).",
    )
    args = parser.parse_args(argv)

    # mlx-whisper convention: verbose=False shows a tqdm progress bar,
    # verbose=True streams per-segment text, verbose=None is silent.
    if args.quiet:
        verbose: bool | None = None
    elif args.show_segments:
        verbose = True
    else:
        verbose = False  # default: progress bar

    files, from_inbox = gather_inputs(args.files)
    if not files:
        where = INBOX if from_inbox else "(no files passed)"
        print(f"nothing to do; {where} is empty.")
        return 0

    cli_prompt_file = (
        Path(args.prompt_file).expanduser() if args.prompt_file else None
    )

    today = dt.date.today().isoformat()
    failures = 0
    print(f"transcribe: {len(files)} file(s) to process; model={args.model}")
    for idx, f in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] {f.name}")
        prompt, prompt_src = resolve_prompt(
            f, args.prompt, cli_prompt_file, args.no_prompt,
        )
        print(f"  prompt: {prompt_src}")
        ok, msg = transcribe_one(
            f, today, args.model,
            from_inbox=from_inbox, verbose=verbose, initial_prompt=prompt,
            use_vad=not args.no_vad,
        )
        if ok:
            print(f"  ✓ {msg}")
        else:
            print(f"  ✗ {msg}", file=sys.stderr)
            failures += 1

    if failures:
        print(f"done: {failures} failure(s).", file=sys.stderr)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
