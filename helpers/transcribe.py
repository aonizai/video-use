"""Transcribe a video with a selectable speech-to-text backend.

Backends:
- local: faster-whisper running on this machine (no API key required)
- elevenlabs: ElevenLabs Scribe (API key required)
- auto: prefer local when faster-whisper is installed, otherwise use ElevenLabs
- none: skip transcription (CLI only; useful for visual-first workflows)

Both transcription backends write a Scribe-compatible word list to
<edit_dir>/transcripts/<video_stem>.json so pack_transcripts.py and the rest
of video-use can keep using the same artifacts.

Cached: if the output file already exists, transcription is skipped unless
--force is supplied.

Usage:
    python helpers/transcribe.py <video_path> --backend local
    python helpers/transcribe.py <video_path> --backend auto
    python helpers/transcribe.py <video_path> --backend elevenlabs
    python helpers/transcribe.py <video_path> --backend none
    python helpers/transcribe.py <video_path> --local-model large-v3
"""

from __future__ import annotations

import argparse
import array
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import requests


SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"
BACKENDS = ("auto", "local", "elevenlabs", "none")


def load_api_key(required: bool = True) -> str | None:
    """Load an ElevenLabs API key from .env or the environment.

    When required=False, return None instead of exiting if no key exists.
    """
    for candidate in [Path(__file__).resolve().parent.parent / ".env", Path(".env")]:
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "ELEVENLABS_API_KEY":
                    value = v.strip().strip('"').strip("'")
                    if value:
                        return value
    value = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if value:
        return value
    if required:
        sys.exit("ELEVENLABS_API_KEY not found in .env or environment")
    return None


def local_backend_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def resolve_backend(backend: str) -> str:
    """Resolve auto to a concrete backend.

    Local-first is deliberate: it lets users run without a paid API and keeps
    interview audio on-device after the model has been downloaded.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown transcription backend: {backend}")
    if backend != "auto":
        return backend
    if local_backend_available():
        return "local"
    if load_api_key(required=False):
        return "elevenlabs"
    raise RuntimeError(
        "No transcription backend is available. Install local STT with "
        '`pip install -e ".[local-stt]"` or configure ELEVENLABS_API_KEY. '
        "For visual-only footage, run with --backend none."
    )


def count_audio_tracks(video_path: Path) -> int:
    """How many audio streams the container holds."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    return len([ln for ln in out.stdout.splitlines() if ln.strip()])


def peak_dbfs(wav_path: Path) -> float:
    """Peak level of a 16-bit PCM wav, in dBFS. -inf for digital silence."""
    peak = 0
    with wave.open(str(wav_path), "rb") as w:
        while frames := w.readframes(1 << 16):
            samples = array.array("h", frames)
            if samples:
                peak = max(peak, max(samples), -min(samples))
    return 20 * math.log10(peak / 32768) if peak > 0 else float("-inf")


def extract_audio(video_path: Path, dest: Path, audio_track: int = 0) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-map", f"0:a:{audio_track}",
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def call_scribe(
    audio_path: Path,
    api_key: str,
    language: str | None = None,
    num_speakers: int | None = None,
) -> dict:
    data: dict[str, str] = {
        "model_id": "scribe_v1",
        "diarize": "true",
        "tag_audio_events": "true",
        "timestamps_granularity": "word",
    }
    if language:
        data["language_code"] = language
    if num_speakers:
        data["num_speakers"] = str(num_speakers)

    with open(audio_path, "rb") as f:
        resp = requests.post(
            SCRIBE_URL,
            headers={"xi-api-key": api_key},
            files={"file": (audio_path.name, f, "audio/wav")},
            data=data,
            timeout=1800,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Scribe returned {resp.status_code}: {resp.text[:500]}")

    payload = resp.json()
    if isinstance(payload, dict):
        payload.setdefault("transcription_backend", "elevenlabs")
    return payload


def call_faster_whisper(
    audio_path: Path,
    language: str | None = None,
    model_name: str = "large-v3",
) -> dict:
    """Run faster-whisper locally and emit a Scribe-compatible word list."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            'Local transcription requires faster-whisper. Install it with '
            '`pip install -e ".[local-stt]"`.'
        ) from exc

    model = _LOCAL_MODEL_CACHE.get(model_name)
    if model is None:
        model = WhisperModel(model_name, device="auto", compute_type="default")
        _LOCAL_MODEL_CACHE[model_name] = model
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )

    words: list[dict] = []
    text_parts: list[str] = []
    for segment in segments:
        segment_text = (getattr(segment, "text", "") or "").strip()
        if segment_text:
            text_parts.append(segment_text)
        for word in getattr(segment, "words", None) or []:
            raw = (getattr(word, "word", "") or "").strip()
            start = getattr(word, "start", None)
            end = getattr(word, "end", None)
            if not raw or start is None or end is None:
                continue
            words.append({
                "text": raw,
                "start": float(start),
                "end": float(end),
                "type": "word",
                # Local mode intentionally does not pretend to diarize.
                "speaker_id": None,
                "probability": float(getattr(word, "probability", 0.0) or 0.0),
            })

    return {
        "text": " ".join(text_parts).strip(),
        "language_code": getattr(info, "language", language),
        "language_probability": float(
            getattr(info, "language_probability", 0.0) or 0.0
        ),
        "transcription_backend": "faster-whisper",
        "model_id": model_name,
        "words": words,
    }


def transcript_path(edit_dir: Path, video: Path, audio_track: int = 0) -> Path:
    """Where a video's transcript lands."""
    suffix = "" if audio_track == 0 else f".track{audio_track}"
    return edit_dir / "transcripts" / f"{video.stem}{suffix}.json"


def transcribe_one(
    video: Path,
    edit_dir: Path,
    api_key: str | None = None,
    language: str | None = None,
    num_speakers: int | None = None,
    verbose: bool = True,
    audio_track: int = 0,
    backend: str = "auto",
    local_model: str = "large-v3",
    force: bool = False,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON."""
    resolved_backend = resolve_backend(backend)
    if resolved_backend == "none":
        raise RuntimeError("transcribe_one cannot write a transcript with backend=none")

    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcript_path(edit_dir, video, audio_track)

    if out_path.exists() and not force:
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    if verbose:
        print(
            f"  extracting audio from {video.name} "
            f"(backend={resolved_backend})",
            flush=True,
        )

    n_tracks = count_audio_tracks(video)
    if n_tracks == 0:
        raise RuntimeError(f"{video.name} has no audio track")
    if audio_track >= n_tracks:
        raise RuntimeError(
            f"{video.name} has {n_tracks} audio track(s); "
            f"--audio-track {audio_track} is out of range"
        )
    if n_tracks > 1 and verbose:
        print(
            f"  note: {video.name} has {n_tracks} audio tracks, using track "
            f"{audio_track + 1} (--audio-track to change)",
            flush=True,
        )

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio, audio_track)

        peak = peak_dbfs(audio)
        if peak < -60.0:
            raise RuntimeError(
                f"track {audio_track + 1} of {video.name} is silent "
                f"(peak {peak:.1f} dBFS). "
                + (
                    "Try a different --audio-track."
                    if n_tracks > 1
                    else "Check the source audio."
                )
            )

        if resolved_backend == "elevenlabs":
            key = api_key or load_api_key(required=True)
            assert key is not None
            size_mb = audio.stat().st_size / (1024 * 1024)
            if verbose:
                print(
                    f"  uploading {video.stem}.wav ({size_mb:.1f} MB) to ElevenLabs",
                    flush=True,
                )
            payload = call_scribe(audio, key, language, num_speakers)
        else:
            if num_speakers and verbose:
                print(
                    "  note: --num-speakers is ignored by the local backend "
                    "(no diarization)",
                    flush=True,
                )
            if verbose:
                print(
                    f"  transcribing locally with faster-whisper model={local_model}",
                    flush=True,
                )
            payload = call_faster_whisper(audio, language, local_model)

    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        if isinstance(payload, dict) and "words" in payload:
            print(f"    words: {len(payload['words'])}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Transcribe a video with local faster-whisper or ElevenLabs Scribe"
    )
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--backend",
        choices=BACKENDS,
        default="auto",
        help=(
            "Transcription backend. auto prefers local faster-whisper, "
            "then ElevenLabs. none skips transcription."
        ),
    )
    ap.add_argument(
        "--local-model",
        default="large-v3",
        help="faster-whisper model name for --backend local (default: large-v3)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code (e.g. th, en). Omit to auto-detect.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional speaker count for ElevenLabs. Ignored by local backend.",
    )
    ap.add_argument(
        "--audio-track",
        type=int,
        default=0,
        help="Zero-based audio track to transcribe.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing cached transcript.",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    if args.backend == "none":
        print("backend=none: transcription skipped")
        return

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    try:
        transcribe_one(
            video=video,
            edit_dir=edit_dir,
            language=args.language,
            num_speakers=args.num_speakers,
            audio_track=args.audio_track,
            backend=args.backend,
            local_model=args.local_model,
            force=args.force,
        )
    except RuntimeError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
