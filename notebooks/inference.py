# %% [markdown]
# # Video Brand Detection — Inference
# Runtime: CPU is fine for short videos, GPU (T4) recommended for longer ones

# %% Setup
# ultralytics without deps so only opencv-python-headless provides cv2
!pip install ultralytics --no-deps --quiet
!pip install "numpy>=1.23.0" "matplotlib>=3.3.0" "opencv-python-headless>=4.7.0" "pillow>=7.1.2" "pyyaml>=5.3.1" "requests>=2.23.0" "torch>=1.8.0" "torchvision>=0.9.0" "psutil>=5.8.0" "polars>=0.20.0" "nvidia-ml-py>=12.0.0" "ultralytics-thop>=2.2.3" "ultralytics-platform>=0.1.45" "huggingface_hub>=0.20" yt-dlp supabase --quiet

# %% Clone repo
import sys
from pathlib import Path

REPO = "mod3-project4-anahi"
REPO_URL = f"https://github.com/Bootcamp-IA-MAD-P7/{REPO}.git"

if not Path(REPO).exists():
    !git clone --depth 1 {REPO_URL}

sys.path.insert(0, str(Path(REPO) / "src"))

# %% Imports
import os

from pipeline import process_video, write_to_supabase

# %% Input
VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # or a local file path

# %% Run pipeline
report = process_video(
    VIDEO_URL,
    frame_interval=5,
    conf_threshold=0.5,
    gap_tolerance=5,
)

# %% Report
print(f"Video: {report['video_name']}")
print(
    f"Duration: {report['duration_seconds']:.1f}s | "
    f"FPS: {report['fps']:.1f} | Frames: {report['total_frames']}"
)

if not report["classes"]:
    print("\nNo brand detections above threshold")

for class_name, class_report in report["classes"].items():
    print(f"\n{class_name}")
    print(
        f"  Total visible: {class_report['total_visible_seconds']:.2f}s "
        f"({class_report['percentage']:.1f}%)"
    )
    for interval in class_report["intervals"]:
        print(
            f"  {interval['start_time']:.2f}s -> {interval['end_time']:.2f}s "
            f"({interval['duration']:.2f}s) "
            f"conf={interval['peak_confidence']:.2f}"
        )
        print(f"  crop: {interval['crop_path']}")

# %% Supabase (optional)
os.environ["SUPABASE_URL"] = ""  # paste from .env
os.environ["SUPABASE_KEY"] = ""  # paste from .env

if os.environ["SUPABASE_URL"] and os.environ["SUPABASE_KEY"]:
    write_to_supabase(report)
    print("Written to Supabase")
else:
    print("Supabase credentials not set, skipping write")
