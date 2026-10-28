# Training Results — YOLOv8s-seg Oreo Detection

## Run

- Model: YOLOv8s-seg (fine-tuned from COCO weights)
- Dataset: Oreo v4 (Roboflow), 425 train / 13 val images after 5x augmentation
- Classes: oreo-cookie, oreo-logo
- Epochs configured: 100
- Early stopping patience: 10
- Best checkpoint: epoch 33
- Training stopped: epoch 43
- Training time: ~0.143 hours on Colab T4
- Weights: Anahia/yolov8s-seg-oreo (Hugging Face Hub, public)

## Loading the Model

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model_path = hf_hub_download(repo_id="Anahia/yolov8s-seg-oreo", filename="best.pt")
model = YOLO(model_path)
```

huggingface_hub downloads the weights on first run and caches them. No manual
download required.

## Validation Metrics at Best Checkpoint

| Class | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|
| all | 0.715 | 0.537 | 0.733 | 0.506 |
| oreo-cookie | 0.855 | 0.729 | 0.855 | 0.712 |
| oreo-logo | 0.574 | 0.345 | 0.611 | 0.299 |

## Analysis

The cookie class performs strongly across both box and mask metrics. The physical
cookie is visually consistent — a round dark wafer — which makes it a reliable
detection target even with 114 source images.

The logo class is significantly weaker, with mask mAP50 of 0.611 and recall of 0.5.
Logo detections miss roughly half of validation instances. Contributing factors:
logos appear at smaller sizes, are more affected by motion blur in video, and vary
more in aspect ratio and crop than the cookie itself. This is acceptable for a proof
of concept but is the primary area to address if detection quality needs to improve.

## Reproducing

1. Open notebooks/train_oreo.py in Google Colab (GPU runtime, T4 or better)
2. Set ROBOFLOW_API_KEY from .env
3. Run all cells
4. Upload best.pt to Anahia/yolov8s-seg-oreo on Hugging Face Hub
