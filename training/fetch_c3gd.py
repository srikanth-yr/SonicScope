"""Download a compact, reproducible gunshot subset from C3GD for training."""
from __future__ import annotations

import argparse
import http.client
import random
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

C3GD_URL = "https://zenodo.org/records/22286299/files/C3GD.zip?download=1"


def download_with_resume(url: str, destination: Path, attempts: int = 8) -> None:
    """Resume a partial C3GD archive instead of discarding a large download."""
    for attempt in range(1, attempts + 1):
        offset = destination.stat().st_size if destination.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                with destination.open(mode) as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                return
        except (http.client.IncompleteRead, OSError) as exc:
            print(f"Download interrupted ({exc}). Retrying {attempt}/{attempts}...")
            time.sleep(min(attempt * 2, 12))
    raise RuntimeError(f"Could not download {url}; partial archive kept at {destination}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--count", type=int, default=240)
    args = parser.parse_args()
    if args.count < 10:
        raise SystemExit("Choose at least 10 gunshot clips.")

    archive = args.data.parent / ".downloads" / "c3gd.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading/resuming the C3GD gunshot dataset (about 771 MB)...")
    download_with_resume(C3GD_URL, archive)
    destination = args.data / "gunshots"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        clips = [name for name in package.namelist() if name.lower().endswith(".wav")]
        if len(clips) < args.count:
            raise RuntimeError(f"C3GD archive contains only {len(clips)} WAV files.")
        selected = random.Random(42).sample(clips, args.count)
        for clip in selected:
            output = destination / f"c3gd_{Path(clip).name}"
            if not output.exists():
                with package.open(clip) as source, output.open("wb") as target:
                    shutil.copyfileobj(source, target)
    print(f"Ready: {len(list(destination.glob('*.wav')))} gunshot clips in {destination}")


if __name__ == "__main__":
    main()
