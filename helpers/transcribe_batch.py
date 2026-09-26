"""Batch-transcribe videos with a selectable speech-to-text backend.

Backends mirror helpers/transcribe.py:
- local: faster-whisper, on-device, no API key
- elevenlabs: ElevenLabs Scribe
- auto: local-first, then ElevenLabs
- none: inventory-friendly no-op

Only video files in the TOP LEVEL of <videos_dir> are considered.

Usage:
    python helpers/transcribe_batch.py <videos_dir> --backend local
    python helpers/transcribe_batch.py <videos_dir> --backend auto
    python helpers/transcribe_batch.py <videos_dir> --backend none
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from transcribe import (
    BACKENDS,
    load_api_key,
    resolve_backend,
    transcribe_one,
    transcript_path,
)


VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV", ".avi", ".AVI", ".m4v"}


def find_videos(videos_dir: Path) -> list[Path]:
    return sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix in VIDEO_EXTS
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel batch transcription of a videos directory")
    ap.add_argument("videos_dir", type=Path, help="Directory containing source videos")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <videos_dir>/edit)",
    )
    ap.add_argument(
        "--backend",
        choices=BACKENDS,
        default="auto",
        help="auto prefers local faster-whisper, then ElevenLabs; none skips transcription",
    )
    ap.add_argument(
        "--local-model",
        default="large-v3",
        help="faster-whisper model name for local backend",
    )
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code. Omit to auto-detect per file.",
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
        help="Overwrite cached transcripts.",
    )
    args = ap.parse_args()

    videos_dir = args.videos_dir.resolve()
    if not videos_dir.is_dir():
        sys.exit(f"not a directory: {videos_dir}")

    videos = find_videos(videos_dir)
    if not videos:
        sys.exit(f"no videos found in {videos_dir}")

    if args.backend == "none":
        print(f"found {len(videos)} videos")
        print("backend=none: transcription skipped")
        return

    try:
        resolved_backend = resolve_backend(args.backend)
    except RuntimeError as exc:
        sys.exit(str(exc))

    edit_dir = (args.edit_dir or (videos_dir / "edit")).resolve()
    (edit_dir / "transcripts").mkdir(parents=True, exist_ok=True)

    already_cached = [
        v for v in videos
        if transcript_path(edit_dir, v, args.audio_track).exists() and not args.force
    ]
    pending = [v for v in videos if v not in already_cached]

    print(
        f"found {len(videos)} videos "
        f"({len(already_cached)} cached, {len(pending)} to transcribe)"
    )
    print(f"backend: {resolved_backend}")
    if not pending:
        print("nothing to do")
        return

    api_key = load_api_key(required=True) if resolved_backend == "elevenlabs" else None

    # Local Whisper models are heavyweight; multiple workers can exhaust RAM/VRAM.
    # Keep local mode sequential unless the user explicitly overrides --workers.
    workers = args.workers
    if resolved_backend == "local" and args.workers == 4:
        workers = 1
        print("local backend: defaulting to 1 worker to avoid loading the model 4 times")

    print(f"transcribing {len(pending)} files with {workers} worker(s)")
    t0 = time.time()

    errors: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                transcribe_one,
                video=v,
                edit_dir=edit_dir,
                api_key=api_key,
                language=args.language,
                num_speakers=args.num_speakers,
                verbose=False,
                audio_track=args.audio_track,
                backend=resolved_backend,
                local_model=args.local_model,
                force=args.force,
            ): v
            for v in pending
        }
        for fut in as_completed(futures):
            v = futures[fut]
            try:
                out = fut.result()
                print(f"  + {v.stem}  →  {out.name}")
            except Exception as exc:
                errors.append((v, str(exc)))
                print(f"  x {v.stem}  FAILED: {exc}")

    dt = time.time() - t0
    print(f"\ndone in {dt:.1f}s")
    if errors:
        print(f"{len(errors)} failures:")
        for v, msg in errors:
            print(f"  {v.name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
