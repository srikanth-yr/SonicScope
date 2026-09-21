from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .audio import decode_pcm16, log_mel, suppress_background_voices
from .model import ModelRunner
from .yamnet import YamnetRunner

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
custom_runner = ModelRunner(ROOT / "models" / "battlefield_sound_cnn.pt")
yamnet_runner = YamnetRunner(ROOT / "models" / "yamnet.tflite", ROOT / "models" / "yamnet_class_map.csv")
DEFAULT_THRESHOLDS = {"footsteps": 0.14, "gunshots": 0.30}
# The custom gunshot set includes short, distant recordings.  Its validation
# calibration selected 0.73, which rejects many genuine short gunshots; 0.60
# remains above every observed background/footstep gunshot score in local tests.
CUSTOM_ALERT_CAPS = {"footsteps": 1.0, "gunshots": 0.60}


def active_runner() -> tuple[str, ModelRunner | YamnetRunner]:
    if custom_runner.ready:
        return "custom", custom_runner
    return "yamnet", yamnet_runner

app = FastAPI(title="Battlefield Sound Sentinel", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/assets", StaticFiles(directory=FRONTEND), name="assets")


class InferenceRequest(BaseModel):
    pcm16: str = Field(description="Base64 little-endian PCM16 mono audio")
    sample_rate: int = Field(ge=8_000, le=96_000)


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
def health() -> dict:
    source, active = active_runner()
    return {
        "status": "ready" if active.ready else "model_unavailable",
        "source": source,
        "detail": active.load_error,
    }


@app.get("/api/demo-audio/{label}")
def demo_audio(label: str) -> FileResponse:
    """Serve one local training sample for the dashboard demonstration controls."""
    if label not in {"footsteps", "gunshots"}:
        raise HTTPException(status_code=404, detail="Unknown demo sound")
    clip = next((ROOT / "data" / "raw" / label).glob("*.wav"), None)
    if clip is None:
        raise HTTPException(status_code=404, detail=f"No {label} demo sound is available")
    return FileResponse(clip, media_type="audio/wav")


@app.post("/api/infer")
def infer(payload: InferenceRequest) -> dict:
    source, active = active_runner()
    if not active.ready:
        raise HTTPException(status_code=503, detail=active.load_error or "Model unavailable")
    try:
        audio = decode_pcm16(payload.pcm16)
        filtered_audio = suppress_background_voices(audio, payload.sample_rate)
        if source == "custom":
            scores = custom_runner.predict(log_mel(filtered_audio, payload.sample_rate))
            voice_score = 0.0
        else:
            scores, voice_score = yamnet_runner.predict(filtered_audio, payload.sample_rate)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    thresholds = {
        label: min(custom_runner.thresholds.get(label, DEFAULT_THRESHOLDS[label]), CUSTOM_ALERT_CAPS[label])
        if source == "custom"
        else DEFAULT_THRESHOLDS[label]
        for label in DEFAULT_THRESHOLDS
    }
    # YAMNet supplies an explicit speech score.  A voice-dominant clip is not an
    # event even if one of the target labels has a weak incidental score.
    voice_dominant = source == "yamnet" and voice_score >= 0.45 and voice_score > max(scores["footsteps"], scores["gunshots"])
    predicted = max(scores, key=scores.get)
    detections = {
        # A secondary class can clear its threshold while background (or the
        # other event) is more probable. Alert only for the top prediction.
        label: bool(predicted == label and scores[label] >= thresholds[label] and not voice_dominant)
        for label in thresholds
    }
    return {
        "detections": detections,
        "source": source,
    }


@app.post("/api/reload-model")
def reload_model() -> dict:
    custom_runner.reload()
    yamnet_runner.reload()
    return health()
