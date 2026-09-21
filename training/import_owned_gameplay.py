"""Create training WAV clips from gameplay recordings you are authorized to use.

Input annotations identify known event times in recordings you own.  This tool
does not download, scrape, or redistribute game audio.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from backend.audio import CLIP_SAMPLES, SAMPLE_RATE
from training.train import load_wav_mono

LABELS = {"background": "background", "footstep": "footsteps", "footsteps": "footsteps", "gunshot": "gunshots", "gunshots": "gunshots"}


def wav_source(path: Path) -> Path:
    """Return a WAV source, using a locally installed ffmpeg only when needed."""
    if path.suffix.lower() == ".wav":
        return path
    descriptor, temporary_name = tempfile.mkstemp(suffix=".wav", prefix="gameplay_audio_")
    os.close(descriptor)
    temporary = Path(temporary_name)
    command = ["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), str(temporary)]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("MP4/MKV input needs ffmpeg. Install it, or provide an exported PCM WAV recording.") from exc
    except subprocess.CalledProcessError as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Could not read {path.name}: {exc.stderr.decode(errors='replace')}") from exc
    return temporary


def export_clip(waveform: torch.Tensor, source_rate: int, start: float, end: float, destination: Path) -> None:
    if source_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, source_rate, SAMPLE_RATE)
    midpoint = (start + end) / 2
    first = round(midpoint * SAMPLE_RATE - CLIP_SAMPLES / 2)
    clip = waveform[max(first, 0) : max(first, 0) + CLIP_SAMPLES]
    left_padding = max(0, -first)
    clip = F.pad(clip, (left_padding, max(0, CLIP_SAMPLES - left_padding - clip.numel())))[:CLIP_SAMPLES]
    pcm = (clip.clamp(-1, 1).numpy() * 32767).astype("<i2")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True, help="CSV with recording,start_seconds,end_seconds,label columns")
    parser.add_argument("--recordings", type=Path, default=Path("data/owned_gameplay"))
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    with args.annotations.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"recording", "start_seconds", "end_seconds", "label"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit("Annotations must contain: recording,start_seconds,end_seconds,label")

    sources: dict[Path, tuple[torch.Tensor, int]] = {}
    temporary_files: list[Path] = []
    added = 0
    try:
        for row_number, row in enumerate(rows, start=2):
            label = LABELS.get(row["label"].strip().lower())
            if not label:
                raise ValueError(f"Row {row_number}: label must be background, footsteps, or gunshots")
            try:
                start, end = float(row["start_seconds"]), float(row["end_seconds"])
            except ValueError as exc:
                raise ValueError(f"Row {row_number}: times must be numeric seconds") from exc
            if start < 0 or end < start:
                raise ValueError(f"Row {row_number}: use non-negative start_seconds <= end_seconds")
            recording = (args.recordings / row["recording"]).resolve()
            if not recording.is_file() or args.recordings.resolve() not in recording.parents:
                raise ValueError(f"Row {row_number}: recording is missing or outside {args.recordings}")
            if recording not in sources:
                source = wav_source(recording)
                if source != recording:
                    temporary_files.append(source)
                sources[recording] = load_wav_mono(source)
            waveform, rate = sources[recording]
            name = f"gameplay_{recording.stem}_{row_number:04d}.wav"
            export_clip(waveform, rate, start, end, args.data / label / name)
            added += 1
    finally:
        for temporary in temporary_files:
            temporary.unlink(missing_ok=True)
    print(f"Imported {added} authorized gameplay clips into {args.data}")


if __name__ == "__main__":
    main()
