from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from ai_edge_litert.interpreter import Interpreter

YAMNET_RATE = 16_000
WINDOW_SAMPLES = 15_600


class YamnetRunner:
    """AudioSet-pretrained YAMNet ONNX classifier for zero-shot baseline detection."""

    def __init__(self, model_path: Path, labels_path: Path) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.interpreter: Interpreter | None = None
        self.input_index: int | None = None
        self.output_index: int | None = None
        self.index: dict[str, int] = {}
        self.load_error: str | None = None
        self.reload()

    @property
    def ready(self) -> bool:
        return self.interpreter is not None

    def reload(self) -> None:
        self.interpreter = None
        self.load_error = None
        if not self.model_path.exists() or not self.labels_path.exists():
            self.load_error = "Pretrained YAMNet files are missing. Run the model setup command."
            return
        try:
            with self.labels_path.open(newline="", encoding="utf-8") as handle:
                self.index = {row[2]: int(row[0]) for row in list(csv.reader(handle))[1:]}
            self.interpreter = Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_index = int(self.interpreter.get_input_details()[0]["index"])
            self.output_index = int(self.interpreter.get_output_details()[0]["index"])
        except (OSError, ValueError, RuntimeError) as exc:
            self.load_error = f"YAMNet could not be loaded: {exc}"

    @staticmethod
    def _windows(audio: np.ndarray, source_rate: int) -> list[np.ndarray]:
        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        if source_rate != YAMNET_RATE:
            waveform = torchaudio.functional.resample(waveform, source_rate, YAMNET_RATE)
        signal = waveform.squeeze(0)
        if signal.numel() <= WINDOW_SAMPLES:
            return [F.pad(signal, (0, WINDOW_SAMPLES - signal.numel())).numpy().astype(np.float32)]
        hop = WINDOW_SAMPLES // 2
        starts = list(range(0, signal.numel() - WINDOW_SAMPLES + 1, hop))
        last_start = signal.numel() - WINDOW_SAMPLES
        if starts[-1] != last_start:
            starts.append(last_start)
        return [signal[start : start + WINDOW_SAMPLES].numpy().astype(np.float32) for start in starts]

    def predict(self, audio: np.ndarray, source_rate: int) -> tuple[dict[str, float], float]:
        if not self.interpreter or self.input_index is None or self.output_index is None:
            raise RuntimeError(self.load_error or "YAMNet is unavailable")
        scores = []
        for window in self._windows(audio, source_rate):
            self.interpreter.set_tensor(self.input_index, window)
            self.interpreter.invoke()
            scores.append(self.interpreter.get_tensor(self.output_index)[0])
        output = np.max(np.stack(scores), axis=0)
        gunshot = output[self.index["Gunshot, gunfire"]]
        footsteps = max(output[self.index[name]] for name in ("Walk, footsteps", "Run"))
        voice = max(output[self.index[name]] for name in ("Speech", "Conversation", "Narration, monologue"))
        background = max(0.0, 1.0 - max(float(gunshot), float(footsteps)))
        return (
            {"background": round(background, 4), "footsteps": round(float(footsteps), 4), "gunshots": round(float(gunshot), 4)},
            round(float(voice), 4),
        )
