from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class SoundEventCNN(nn.Module):
    """Compact CNN designed for 2-second, 64-bin log-mel inputs."""

    def __init__(self, class_count: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 96), nn.ReLU(),
            nn.Dropout(0.25), nn.Linear(96, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ModelRunner:
    def __init__(self, model_path: str | Path = "models/battlefield_sound_cnn.pt") -> None:
        self.model_path = Path(model_path)
        self.model: SoundEventCNN | None = None
        self.classes: list[str] = []
        self.thresholds: dict[str, float] = {}
        self.load_error: str | None = None
        self.reload()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def reload(self) -> None:
        self.model = None
        self.load_error = None
        if not self.model_path.exists():
            self.load_error = f"Checkpoint not found: {self.model_path}"
            return
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
            self.classes = checkpoint["classes"]
            self.thresholds = checkpoint.get("thresholds", {})
            self.model = SoundEventCNN(len(self.classes))
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
        except (KeyError, RuntimeError, OSError) as exc:
            self.load_error = f"Checkpoint could not be loaded: {exc}"
            self.model = None

    @torch.inference_mode()
    def predict(self, feature: torch.Tensor) -> dict[str, float]:
        if not self.model:
            raise RuntimeError(self.load_error or "No model loaded")
        probabilities = torch.softmax(self.model(feature.unsqueeze(0)), dim=1)[0]
        return {label: round(float(score), 4) for label, score in zip(self.classes, probabilities)}
