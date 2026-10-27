# Oreo Brand Detection

Computer vision pipeline that detects Oreo product placements in video content and reports how long the brand appears on screen.

Built as a proof of concept for an advertising agency use case: given a video, identify every appearance of a brand and produce a time report.

## What it does

- Detects two classes: oreo_cookie and oreo_logo using a YOLOv8s segmentation model
- Processes a video from a URL (YouTube, Vimeo, Instagram, and hundreds of other
  platforms via yt-dlp) or a locally uploaded file
- Groups consecutive detections into appearance intervals
- Stores intervals in a PostgreSQL database (Supabase)
- Saves one representative crop per interval (highest confidence frame)
- Generates a report: total time visible, percentage of video, timestamps
- Exposes a Gradio interface to paste a video URL or upload a file directly

## Stack

- Python 3.13
- uv (dependency management)
- YOLOv8s-seg (ultralytics)
- Model weights: Hugging Face Hub (Anahia/yolov8s-seg-oreo, auto-downloaded on first run)
- OpenCV
- PostgreSQL via Supabase
- Gradio
- Docker

## Project structure

brand-detection/
├── src/
│   ├── detect.py
│   ├── database.py
│   ├── report.py
│   └── app.py
├── notebooks/
│   └── train_oreo.py    # dataset download + training
├── outputs/
│   └── crops/            
├── docs/
│   ├── specs.md
│   ├── architecture.md
│   └── train.md
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml

## Setup

    cd brand-detection
    cp .env.example .env
    uv sync

Fill in your Supabase credentials in .env before running anything.

## Training

Training runs on Google Colab using a free T4 GPU.

1. Open notebooks/train_oreo.py in Colab
2. Run cells in order — dataset downloads from Roboflow, model trains, best.pt is saved
3. Download best.pt from Colab
4. Upload best.pt to Hugging Face

Training hyperparameters: epochs=100, patience=10 (early stopping), batch=16, image size=640.

## Run locally

    uv run python src/detect.py --video path/to/video.mp4

## Run with Docker

    docker-compose up

The image does not include weights — they are downloaded from Hugging Face Hub on startup.

## Deployment

The MVP runs in two environments: Google Colab for the public Gradio demo (share=True) and local Docker with Gradio for development. Both load the same weights from Hugging Face Hub.

## Demo

The Gradio interface runs on Google Colab with a public URL — no deployment needed.

1. Open notebooks/train_oreo.py in Colab
2. Skip to the demo section
3. Run the Gradio cell — a public gradio.live URL is printed automatically
4. Paste a video URL or upload a file, then view the detection report

## Environment variables

    SUPABASE_URL=
    SUPABASE_KEY=
    DATABASE_URL=