from __future__ import annotations

import argparse
import copy
import json
import random
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import precision_recall_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset

from backend.audio import CLIP_SAMPLES, SAMPLE_RATE, log_mel
from backend.model import SoundEventCNN

CLASSES = ["background", "footsteps", "gunshots"]


def load_wav_mono(path: Path) -> tuple[torch.Tensor, int]:
    """Load a PCM WAV without relying on TorchAudio's optional TorchCodec package."""
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if channels < 1 or sample_width not in {1, 2, 3, 4}:
        raise ValueError(f"Unsupported WAV format in {path}")
    if sample_width == 1:
        values = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        values = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        packed = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        values = (packed[:, 0].astype(np.int32) | (packed[:, 1].astype(np.int32) << 8) | (packed[:, 2].astype(np.int32) << 16))
        values = ((values ^ 0x800000) - 0x800000).astype(np.float32) / 8388608.0
    else:
        values = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    return torch.from_numpy(values.reshape(-1, channels).mean(axis=1).copy()), sample_rate


class AudioDataset(Dataset):
    def __init__(self, files: list[tuple[Path, int]], training: bool) -> None:
        self.files, self.training = files, training

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.files[index]
        waveform, sample_rate = load_wav_mono(path)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
        if waveform.numel() > CLIP_SAMPLES:
            start = random.randint(0, waveform.numel() - CLIP_SAMPLES) if self.training else 0
            waveform = waveform[start : start + CLIP_SAMPLES]
        else:
            waveform = F.pad(waveform, (0, CLIP_SAMPLES - waveform.numel()))
        if self.training:
            gain = random.uniform(0.70, 1.25)
            waveform = (waveform * gain).clamp(-1, 1)
            if random.random() < 0.35:
                waveform = (waveform + 0.008 * torch.randn_like(waveform)).clamp(-1, 1)
        feature = log_mel(waveform.numpy(), SAMPLE_RATE)
        if self.training:
            # Masking frequency and time bands makes the classifier less tied to one microphone or surface.
            if random.random() < 0.5:
                width = random.randint(6, 24)
                start = random.randint(0, feature.shape[-1] - width)
                feature[:, start : start + width] = 0
            if random.random() < 0.4:
                width = random.randint(3, 10)
                start = random.randint(0, feature.shape[-2] - width)
                feature[start : start + width, :] = 0
        return feature, label


def collect_files(data_dir: Path) -> list[tuple[Path, int]]:
    files: list[tuple[Path, int]] = []
    for label, name in enumerate(CLASSES):
        files.extend((path, label) for path in (data_dir / name).rglob("*.wav"))
    return files


def class_thresholds(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    targets = labels.cpu().numpy()
    thresholds = {}
    for index, name in enumerate(CLASSES):
        if name == "background":
            continue
        precision, recall, threshold = precision_recall_curve(targets == index, probs[:, index])
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        choice = min(int(f1.argmax()), len(threshold) - 1)
        thresholds[name] = round(float(threshold[choice]), 3)
    return thresholds


def stratified_split(files: list[tuple[Path, int]], validation_fraction: float = 0.2) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Keep every event class represented in both training and validation."""
    grouped: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for item in files:
        grouped[item[1]].append(item)
    train_files, validation_files = [], []
    randomizer = random.Random(42)
    for label, items in grouped.items():
        randomizer.shuffle(items)
        split = max(1, min(len(items) - 1, round(len(items) * validation_fraction)))
        validation_files.extend(items[:split])
        train_files.extend(items[split:])
    randomizer.shuffle(train_files)
    randomizer.shuffle(validation_files)
    return train_files, validation_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("models/battlefield_sound_cnn.pt"))
    args = parser.parse_args()

    files = collect_files(args.data)
    counts = defaultdict(int)
    for _, label in files:
        counts[label] += 1
    if len(files) < 30 or any(counts[i] < 10 for i in range(len(CLASSES))):
        raise SystemExit("Add at least 10 WAV files per class (30 total) before training.")
    train_files, val_files = stratified_split(files)
    train_loader = DataLoader(AudioDataset(train_files, True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(AudioDataset(val_files, False), batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SoundEventCNN(len(CLASSES)).to(device)
    weight = torch.tensor([1 / max(counts[i], 1) for i in range(len(CLASSES))], device=device)
    criterion = nn.CrossEntropyLoss(weight=weight / weight.sum() * len(CLASSES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss, best = float("inf"), None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, labels in train_loader:
            logits = model(features.to(device))
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        val_logits, val_labels, losses = [], [], []
        with torch.inference_mode():
            for features, labels in val_loader:
                logits = model(features.to(device))
                losses.append(float(criterion(logits, labels.to(device))))
                val_logits.append(logits.cpu())
                val_labels.append(labels)
        loss = sum(losses) / len(losses)
        accuracy = (torch.cat(val_logits).argmax(1) == torch.cat(val_labels)).float().mean().item()
        print(f"epoch {epoch:02d} | validation loss {loss:.4f} | accuracy {accuracy:.1%}")
        if loss < best_loss:
            best_loss = loss
            # state_dict tensors share storage with the live model unless copied.
            # Keep the exact epoch that produced the selected validation logits.
            best = (copy.deepcopy(model.state_dict()), torch.cat(val_logits), torch.cat(val_labels), accuracy)

    assert best is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": best[0], "classes": CLASSES,
        "thresholds": class_thresholds(best[1], best[2]),
        "validation_accuracy": round(best[3], 4),
    }
    torch.save(checkpoint, args.output)
    args.output.with_suffix(".json").write_text(json.dumps({k: v for k, v in checkpoint.items() if k != "model_state"}, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
