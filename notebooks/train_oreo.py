# %% [markdown]
# # YOLOv8 Training — Oreo Detection
# Runtime: GPU (T4 recommended)

# %% Setup
!nvidia-smi
!pip install ultralytics roboflow --quiet

# %% Imports
from pathlib import Path
from roboflow import Roboflow
from ultralytics import YOLO

# %% Dataset
api_key = ""  # paste ROBOFLOW_API_KEY from .env

rf = Roboflow(api_key=api_key)
project = rf.workspace("anahi-amerise").project("oreo-v4aez")
version = project.version(4)
dataset = version.download("yolov8-seg")

DATASET_PATH = Path(dataset.location)
DATA_YAML = DATASET_PATH / "data.yaml"

# %% Train
model = YOLO("yolov8s-seg.pt")

results = model.train(
    data=str(DATA_YAML),
    epochs=100,
    patience=10,
    imgsz=640,
    batch=16,
    project="oreo",
    name="v1",
    exist_ok=True,
)

# %% Validate
metrics = model.val()
print(metrics)

# %% Inference (sample)
sample_images = list((DATASET_PATH / "test" / "images").glob("*.jpg"))[:5]

for img in sample_images:
    model.predict(str(img), save=True, conf=0.25)

# %%
from google.colab import files
files.download("/content/runs/segment/oreo/v1/weights/best.pt")