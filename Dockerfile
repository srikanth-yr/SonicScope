FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY models/battlefield_sound_cnn.pt models/battlefield_sound_cnn.json ./models/
COPY models/yamnet.tflite models/yamnet_class_map.csv ./models/
COPY data/raw/footsteps/esc50_1-155858-A-25.wav ./data/raw/footsteps/
COPY data/raw/gunshots/sesa_gunshot_000.wav ./data/raw/gunshots/

EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
