# Architecture Decisions

## Model: YOLOv8 over YOLO26

YOLO26 (released January 2026) outperforms YOLOv8 across all metrics at comparable model sizes and uses fewer parameters. However, research shows YOLOv8 achieves 90% mAP50 with as few as 400 training images, while YOLO26 nano and small variants required 1,000 images to reach comparable accuracy.

This project has 114 training images by design — a small, focused dataset for a proof of concept. At that scale, YOLOv8 is the correct choice. YOLO26's architectural improvements do not compensate for its higher data requirements when the dataset is this small.

Additionally, YOLOv8 has significantly more community resources, solved problems, and Colab examples — relevant when debugging unfamiliar tooling for the first time.

Both models use the identical ultralytics Python API, so switching to YOLO26 in a future iteration with a larger dataset costs one line of code.

## Model size: small (s) over nano (n)

YOLOv8n is used first to verify the pipeline runs end to end without errors. YOLOv8s is the actual training target. It has enough capacity to learn logo shapes reliably without overfitting a small dataset. Larger variants (m, l, x) would be overparameterized for 114 images and offer no accuracy benefit.

## Early stopping with patience=10

Training runs for up to 100 epochs but stops after 10 epochs without validation improvement. On a small dataset of 114 images, the model converges well before epoch 100 and further training only risks overfitting. Early stopping keeps training time short on the free Colab T4 GPU while still allowing the run to reach its best checkpoint.

## Segmentation over detection

Chose YOLOv8s-seg over YOLOv8s (detection) to produce pixel-level masks rather than bounding boxes. The trade-off is slightly slower inference and stricter annotation requirements. Acceptable given the controlled demo environment, the educational goal of the project, and the fact that polygon annotations were already collected in Roboflow.

## Two classes: oreo_cookie and oreo_logo

Initially considered one class (oreo) covering both the physical cookie and the logo. Split into two classes to give the report more granularity and to make the demo more technically interesting. Trade-off: 114 images total, which is thin for segmentation. Compensated by augmentation.

## Appearance intervals over per-frame storage

Storing every detected frame produces thousands of DB rows per video with no added reporting value. Grouping consecutive detections into intervals (appeared at 3.2s, disappeared at 7.8s) maps directly to what the client needs and reduces DB writes significantly. Gap tolerance between frames is configurable.

## No cutout augmentation

Cutout exists to simulate occlusion when you have no real examples of it. This dataset already contains actual cropped and partially-visible objects, which are a higher quality occlusion signal than a synthetic black rectangle placed randomly: real partial crops have realistic edges, context, and proportions, while a cutout box is a blunt approximation. The augmentation set (flip, rotation, brightness, blur, noise) covers photometric and geometric variation without it.

## One crop per interval

One representative crop per appearance interval — the frame with the highest confidence score within that interval — is stored. This provides visual verification of detections without the storage overhead of saving every frame.

## Supabase over SQLite

SQLite requires no setup and is simpler, but Supabase provides a real PostgreSQL instance accessible from Colab, local development, and Docker without managing a local DB file. Better suited for a project that runs across multiple environments. Also a more portfolio-relevant technology choice.

## Gradio over Streamlit

Gradio has native Colab support via share=True, which generates a public gradio.live URL without any additional tooling. Streamlit requires ngrok to expose a public URL from Colab — an extra dependency, extra account, extra configuration. Since the demo runs on Colab, Gradio is the simpler and more reliable choice.

## Video input: URL and file upload

The Gradio interface accepts both a video URL and a direct file upload. URL downloads are handled by yt-dlp, which supports hundreds of platforms
(YouTube, Vimeo, Instagram, Twitter, etc.) without platform-specific code. File upload is included as a fallback for local files or platforms yt-dlp does not support. Both inputs feed the same inference pipeline — the URL
path simply adds a download step before processing.

## Training on Colab, weights not committed

Training is a one-off development step, not part of the application runtime. The notebook lives in notebooks/train.py as a .py file with # %% cells,
runnable in Colab as a notebook. best.pt is downloaded manually after training and placed at models/weights/, which is gitignored. This keeps the repository lightweight and the training environment separate from the application.

## One model, multiple classes

An early alternative was to train a separate model per brand and run them in sequence. Rejected: running N models for N brands is wasteful, slow, and harder to maintain. One multi-class model is the correct pattern — retrained when a new brand is added. The current single-brand implementation already follows this architecture.

## uv over pip / poetry

uv is significantly faster than pip for dependency resolution and installation, and simpler than poetry for a project of this scale. pyproject.toml is the single source of truth for dependencies.

## Frame skipping

Processing every frame on CPU is not feasible in reasonable time. Every 5th frame is processed by default, reducing inference time by 80% with minimal impact on detection accuracy — logos do not disappear and reappear frame by frame. The interval is configurable per video.

## Python 3.13

Ultralytics fixed Python 3.13 pathlib and checkpoint compatibility in v8.4.99 (July 2026). Python 3.13 is safe to use and is the current stable release, so it is the natural choice.