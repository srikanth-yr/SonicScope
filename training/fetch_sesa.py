"""Fetch the compact MIT-licensed SESA gunshot subset from Hugging Face."""
from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

API_URL = "https://huggingface.co/api/datasets/MahiA/SESA/tree/main/audios?recursive=true&expand=false"
DOWNLOAD_ROOT = "https://huggingface.co/datasets/MahiA/SESA/resolve/main/"


def fetch(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "sound-sentinel-training/1.0"})
    with urllib.request.urlopen(request, timeout=60) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    destination = args.data / "gunshots"
    destination.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API_URL, timeout=60) as response:
        files = json.load(response)
    clips = sorted(item["path"] for item in files if item["path"].startswith("audios/gunshot_") and item["path"].endswith(".wav"))
    if len(clips) < 10:
        raise RuntimeError("SESA repository did not return enough gunshot clips.")
    for index, clip in enumerate(clips, start=1):
        output = destination / f"sesa_{Path(clip).name}"
        if not output.exists():
            print(f"Downloading gunshot {index}/{len(clips)}...", flush=True)
            fetch(DOWNLOAD_ROOT + clip, output)
    print(f"Ready: {len(list(destination.glob('*.wav')))} gunshot clips in {destination}")


if __name__ == "__main__":
    main()
