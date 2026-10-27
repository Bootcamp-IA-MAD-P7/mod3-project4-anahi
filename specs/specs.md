# Project Specifications

## Problem

An advertising agency needs to measure brand visibility in video content.
Given a video, the system must detect when a brand appears and report
total visibility time and percentage.

## Scope

Proof of concept. One brand (Oreo). Architecture supports adding more brands
by retraining with additional classes.

## Dataset

- Images: 114, annotations: 275, classes: 2 (oreo_cooki, oreo_logo)
- oreo_cookie: 114 train / 21 val / 23 test (158 annotations total)
- oreo_logo: 77 train / 20 val / 20 test (117 annotations total)
- Annotation type: polygon segmentation (Roboflow)
- Augmentations: flip horizontal, rotation ±15°, brightness ±20%, blur up to 2px, noise up to 2%, multiplier: 5x
- Export format: YOLOv8 segmentation

## Model

- Architecture: YOLOv8s-seg (ultralytics)
- Pre-trained weights: yolov8s-seg.pt (COCO)
- Fine-tuned on: Oreo dataset
- Training environment: Google Colab (T4 GPU)
- Epochs: 100
- Early stopping patience: 10
- Image size: 640
- Batch size: 16

## Video Processing

- Input: downloaded video file (MP4)
- Frame sampling: every 5th frame (configurable)
- Confidence threshold: 0.5 (configurable)
- Max video length: ~1 minute for demo

## Detection Logic

Consecutive detections of the same class within a gap tolerance of N frames
are grouped into a single appearance interval.

Each interval stores:
- start_timestamp (seconds)
- end_timestamp (seconds)
- duration (seconds)
- brand class
- peak confidence score
- one representative crop (highest confidence frame)

## Database

- Provider: Supabase (hosted PostgreSQL)
- Connection: DATABASE_URL in .env

Schema:

    detections
    ├── id                  SERIAL PRIMARY KEY
    ├── video_name          TEXT
    ├── brand               TEXT
    ├── start_time          FLOAT
    ├── end_time            FLOAT
    ├── duration            FLOAT
    ├── peak_confidence     FLOAT
    ├── crop_path           TEXT
    └── created_at          TIMESTAMP DEFAULT NOW()

    videos
    ├── id                  SERIAL PRIMARY KEY
    ├── video_name          TEXT
    ├── duration_seconds    FLOAT
    ├── fps                 FLOAT
    ├── processed_at        TIMESTAMP DEFAULT NOW()
    └── total_frames        INTEGER

## Report Output

Per video, per class:
- Total seconds visible
- Percentage of video duration
- List of appearance intervals with timestamps

## Interface

- Gradio app
- Input: video file upload
- Output: report displayed in UI, crops shown per interval
- Deployment: Google Colab with share=True for demo

## Delivery Levels Targeted

- Essential: model detects one brand with bounding mask visible ✓
- Medium: works on video, class name shown ✓
- Advanced: confidence score shown, DB storage, multi-class architecture ✓
- Expert: Gradio UI ✓ / cloud deploy ✗

## Infrastructure

- Containerized with Docker
- Environment variables via .env
- Dependencies managed with uv
- Python 3.13