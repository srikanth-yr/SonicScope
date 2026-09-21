from __future__ import annotations

import base64
import binascii

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

SAMPLE_RATE = 16_000
CLIP_SECONDS = 2.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
N_MELS = 64
N_FRAMES = 201

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=512,
    hop_length=160,
    n_mels=N_MELS,
    f_min=30,
    f_max=7_600,
)


def suppress_background_voices(audio: np.ndarray, source_rate: int) -> np.ndarray:
    """Reduce sustained voice/background energy while preserving short transients.

    A percentile noise estimate is subtracted in the STFT domain and converted to
    a Wiener-style gain.  Frames with high spectral flux retain most of their
    energy, which protects short impact sounds such as footsteps and gunshots.
    The remaining, non-transient speech band receives a modest extra reduction.
    This is deliberately conservative: voice and target sounds can overlap in
    frequency, so a hard band-stop would also remove useful event information.
    """
    if source_rate <= 0:
        raise ValueError("Audio sample rate must be positive")

    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).flatten()
    if waveform.numel() < 2:
        return waveform.numpy()

    # A 32 ms analysis window gives enough frequency resolution for speech while
    # keeping impulsive sounds local in time.  A power of two is efficient for FFT.
    target_window = max(256, int(source_rate * 0.032))
    n_fft = 1 << (target_window - 1).bit_length()
    hop_length = n_fft // 4
    original_length = waveform.numel()
    if original_length < n_fft:
        waveform = F.pad(waveform, (0, n_fft - original_length))

    window = torch.hann_window(n_fft, dtype=waveform.dtype)
    spectrum = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        return_complex=True,
    )
    magnitude = spectrum.abs()
    epsilon = torch.finfo(magnitude.dtype).eps

    # The lower percentile is a robust estimate of the persistent background in
    # each frequency bin; unlike a mean, intermittent footsteps do not inflate it.
    noise_floor = torch.quantile(magnitude, 0.20, dim=1, keepdim=True)
    residual = (magnitude - 1.15 * noise_floor).clamp_min(0.0)
    gain = (residual / (magnitude + epsilon)).clamp(0.12, 1.0)

    frame_energy = magnitude.mean(dim=0)
    flux = torch.zeros_like(frame_energy)
    flux[1:] = (frame_energy[1:] - frame_energy[:-1]).abs() / (frame_energy[:-1] + epsilon)
    transient = flux >= torch.quantile(flux, 0.75)

    # Most speech energy is in this band.  Do not apply this additional gain to
    # transient frames so impacts retain their broadband signature.
    frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / source_rate)
    speech_band = (frequencies >= 180) & (frequencies <= 3_800)
    speech_gain = torch.where(transient, torch.ones_like(frame_energy), torch.full_like(frame_energy, 0.78))
    gain[speech_band] *= speech_gain.unsqueeze(0)

    filtered = torch.istft(
        spectrum * gain,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        length=waveform.numel(),
    )
    return filtered[:original_length].numpy().astype(np.float32, copy=False)


def decode_pcm16(encoded_pcm: str) -> np.ndarray:
    """Decode base64 little-endian mono PCM from the browser."""
    try:
        data = base64.b64decode(encoded_pcm, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Audio payload must be valid base64 PCM16") from exc
    if len(data) % 2:
        raise ValueError("PCM16 payload must have an even byte length")
    audio = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if not len(audio):
        raise ValueError("Audio payload is empty")
    return audio


def log_mel(audio: np.ndarray, source_rate: int) -> torch.Tensor:
    """Return normalized [1, 64, 201] log-mel input expected by the model."""
    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    if source_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, source_rate, SAMPLE_RATE)
    waveform = waveform.mean(dim=0, keepdim=True)
    waveform = F.pad(waveform, (0, max(0, CLIP_SAMPLES - waveform.shape[-1])))
    waveform = waveform[:, :CLIP_SAMPLES]
    mel = _mel(waveform).clamp_min(1e-8).log()
    mel = F.pad(mel, (0, max(0, N_FRAMES - mel.shape[-1])))[:, :, :N_FRAMES]
    return (mel - mel.mean()) / (mel.std().clamp_min(1e-6))
