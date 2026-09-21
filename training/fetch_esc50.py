"""Download curated ESC-50 background and footstep samples.

ESC-50 contains a footstep category but no gunshot category. Use
``training.fetch_sesa`` or ``training.fetch_c3gd`` for gunshot clips.
"""
from __future__ import annotations

import argparse
import csv
import http.client
import random
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

ARCHIVE_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"


def download_with_resume(url: str, destination: Path, attempts: int = 8) -> None:
    """Resume a partial archive when GitHub closes a long transfer early."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        offset = destination.stat().st_size if destination.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                # A server that ignores Range returns the whole archive, so replace the partial file.
                mode = "ab" if offset and response.status == 206 else "wb"
                with destination.open(mode) as output:
                    while True:
                        try:
                            chunk = response.read(1024 * 1024)
                        except http.client.IncompleteRead as exc:
                            if exc.partial:
                                output.write(exc.partial)
                            raise
                        if not chunk:
                            return
                        output.write(chunk)
        except (http.client.IncompleteRead, OSError) as exc:
            print(f"Transfer interrupted ({exc}). Retrying {attempt}/{attempts}...")
            time.sleep(min(attempt * 2, 12))
    raise RuntimeError(f"Could not download {url} after {attempts} attempts. Partial file kept at {destination}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--background-count", type=int, default=160)
    args = parser.parse_args()

    archive = args.data.parent / ".downloads" / "esc50.zip"
    print("Downloading ESC-50 archive (resumes automatically if interrupted)...")
    download_with_resume(ARCHIVE_URL, archive)
    try:
        with zipfile.ZipFile(archive) as package:
            metadata = next(name for name in package.namelist() if name.endswith("meta/esc50.csv"))
            rows = list(csv.DictReader(line.decode("utf-8") for line in package.open(metadata)))
            target_rows = [row for row in rows if row["category"] == "footsteps"]
            other_rows = [row for row in rows if row["category"] != "footsteps"]
            random.Random(42).shuffle(other_rows)
            selected = [(row, row["category"]) for row in target_rows]
            selected.extend((row, "background") for row in other_rows[: args.background_count])
            for category in ("background", "footsteps", "gunshots"):
                (args.data / category).mkdir(parents=True, exist_ok=True)
            for row, category in selected:
                output_category = "gunshots" if category == "gunshot" else category
                source = next(name for name in package.namelist() if name.endswith(f"audio/{row['filename']}"))
                destination = args.data / output_category / f"esc50_{row['filename']}"
                with package.open(source) as original, destination.open("wb") as output:
                    shutil.copyfileobj(original, output)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Downloaded archive is incomplete or invalid: {archive}. Run the command again to resume it.") from exc
    print(f"Added {len(selected)} ESC-50 samples to {args.data}")


if __name__ == "__main__":
    main()
