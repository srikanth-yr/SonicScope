"""Quickly evaluate the saved checkpoint against the local WAV folders."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from backend.audio import log_mel, suppress_background_voices
from backend.model import ModelRunner
from training.train import CLASSES, load_wav_mono


def main() -> None:
    runner = ModelRunner()
    if not runner.ready:
        raise SystemExit(runner.load_error or "Checkpoint unavailable")
    for expected in CLASSES:
        counts = Counter()
        accepted = 0
        gunshot_scores: list[float] = []
        files = list((Path("data/raw") / expected).glob("*.wav"))
        for path in files:
            waveform, rate = load_wav_mono(path)
            filtered = suppress_background_voices(waveform.numpy(), rate)
            scores = runner.predict(log_mel(filtered, rate))
            gunshot_scores.append(scores["gunshots"])
            predicted = max(scores, key=scores.get)
            counts[predicted] += 1
            accepted += predicted == expected and scores.get(expected, 0.0) >= runner.thresholds.get(expected, 1.0)
        p50, p90, maximum = np.quantile(gunshot_scores, [0.50, 0.90, 1.0]).tolist()
        print(
            f"{expected}: {len(files)} clips | predictions {dict(counts)} | "
            f"above alert threshold {accepted}/{len(files)} | "
            f"gunshot score p50/p90/max {p50:.3f}/{p90:.3f}/{maximum:.3f}"
        )


if __name__ == "__main__":
    main()
