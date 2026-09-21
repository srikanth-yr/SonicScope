# Battlefield Sound Sentinel

Local, real-time acoustic event detection for **footsteps** and **gunshots**. The system has a browser dashboard, a FastAPI inference service, and a PyTorch training pipeline.

> Important: model accuracy depends on representative, legally collected audio and field validation. Do not use an unvalidated model as the sole basis for safety-critical decisions.

## Quick start

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Start the dashboard with the bundled AudioSet-pretrained YAMNet baseline:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000
   ```

   It uses the model in `models/yamnet.tflite` automatically.

3. For a site-specific model, add WAV files to `data/raw/<class>/` and train. The bundled ESC-50 archive supplies background and footsteps only; fetch the compact SESA gunshot subset before the first training run:

   ```powershell
   .\.venv\Scripts\python.exe -m training.fetch_sesa
   .\.venv\Scripts\python.exe -m training.train --epochs 60
   ```

   For a larger production-oriented gunshot set, `training.fetch_c3gd` can download and sample the C3GD dataset separately.

   To improve detection of a specific game, use recordings you own or are licensed to use. Put WAV recordings (or MP4/MKV recordings if `ffmpeg` is installed) in `data/owned_gameplay/`, copy `training/gameplay_annotations.example.csv`, and label known event times. Then import and retrain:

   ```powershell
   .\.venv\Scripts\python.exe -m training.import_owned_gameplay --annotations data\pubg_annotations.csv
   .\.venv\Scripts\python.exe -m training.train --epochs 60
   ```

   Include at least 50 clips each for gunshots and footsteps, plus background clips from the same maps, headphones, volume settings, and recording setup. Keep recordings from one gameplay session in the same split when evaluating a production model.

   For an initial public-data baseline, download the curated ESC-50 subset first:

   ```powershell
   .\.venv\Scripts\python.exe -m training.fetch_esc50
   ```

4. Restart the dashboard after training:

   ```powershell
   uvicorn backend.main:app --reload --port 8000
   ```

4. Open `http://127.0.0.1:8000` and select **Start monitoring**. Your browser will ask for microphone access.

## Phone deployment

The project includes `Dockerfile` and `railway.json` for a public HTTPS deployment. The image includes the trained checkpoint and the two dashboard demo sounds, but not the full training dataset. Deploy it to Railway from a repository containing the tracked files, then use Railway's generated HTTPS domain on your phone. HTTPS is required for mobile browsers to grant microphone access.

## Dataset layout

```
data/raw/
  background/   # wind, engines, voices, blasts, rain, silence, etc.
  footsteps/
  gunshots/
```

Use several recording devices, distances, surfaces, weather conditions, and noise levels. Keep recordings from the same capture session in only one split to prevent optimistic validation scores.

## System layout

```
backend/       API, feature extraction, checkpoint inference
frontend/      live monitoring dashboard
training/      augmentation and model training
models/        generated checkpoint and class metadata
data/          local training audio (ignored by Git)
```

The dashboard announces each confirmed event immediately with a short beep followed by browser speech synthesis ("Footsteps detected" or "Gunshots detected"). A per-class cooldown prevents repeated alerts; confidence values are not displayed. Before inference, an adaptive spectral filter estimates persistent background energy, uses a Wiener-style gain, preserves high-flux transients, and softly attenuates the non-transient speech band. This reduces background voices but cannot guarantee complete voice removal where speech overlaps an event; include voice-heavy background recordings when training the site-specific model. YAMNet is a general AudioSet baseline; the custom checkpoint takes priority when present and is the recommended path for verified deployment accuracy.
