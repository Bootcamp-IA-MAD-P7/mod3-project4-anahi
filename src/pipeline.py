import os
import re
from collections import defaultdict
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import cv2
import yt_dlp
from huggingface_hub import hf_hub_download
from supabase import create_client
from ultralytics import YOLO

WEIGHTS_REPO = "Anahia/yolov8s-seg-oreo"
WEIGHTS_FILENAME = "best.pt"
DEFAULT_FRAME_INTERVAL = 5
DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_GAP_TOLERANCE = 5
DEFAULT_BATCH_SIZE = 8
TMP_DIR = Path("/tmp")
CROPS_ROOT = TMP_DIR / "crops"
STORAGE_BUCKET = "crops"


def resolve_video_path(source: str) -> Path:
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        return _download_video(source)
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    return path


def _download_video(url: str) -> Path:
    ydl_opts = {
        "outtmpl": str(TMP_DIR / "%(title).50s.%(ext)s"),
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    if not path.exists() and path.with_suffix(".mp4").exists():
        path = path.with_suffix(".mp4")
    if not path.is_file():
        raise RuntimeError(f"Download failed for: {url}")
    return path


def read_video_metadata(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or total_frames <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": total_frames / fps,
    }


def detect_frames(
    model: YOLO,
    video_path: Path,
    metadata: dict,
    frame_interval: int,
    conf_threshold: float,
) -> list[dict]:
    detections = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        for frame_index in range(0, metadata["total_frames"], frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            timestamp = frame_index / metadata["fps"]
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    detections.append(
                        {
                            "frame_index": frame_index,
                            "timestamp": timestamp,
                            "class_name": model.names[int(box.cls[0])],
                            "confidence": float(box.conf[0]),
                            "bbox": (x1, y1, x2, y2),
                        }
                    )
    finally:
        cap.release()
    return detections


def group_intervals(detections: list[dict], gap_tolerance: int) -> list[dict]:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for det in sorted(detections, key=lambda d: d["frame_index"]):
        by_class[det["class_name"]].append(det)

    intervals = []
    for class_name, class_detections in by_class.items():
        current = None
        for det in class_detections:
            if (
                current is not None
                and det["frame_index"] - current["last_frame_index"] > gap_tolerance
            ):
                intervals.append(current)
                current = None
            if current is None:
                current = {
                    "class_name": class_name,
                    "first_frame_index": det["frame_index"],
                    "last_frame_index": det["frame_index"],
                    "start_timestamp": det["timestamp"],
                    "end_timestamp": det["timestamp"],
                    "peak_confidence": det["confidence"],
                    "peak_frame_index": det["frame_index"],
                    "peak_bbox": det["bbox"],
                }
            else:
                current["last_frame_index"] = det["frame_index"]
                current["end_timestamp"] = det["timestamp"]
                if det["confidence"] > current["peak_confidence"]:
                    current["peak_confidence"] = det["confidence"]
                    current["peak_frame_index"] = det["frame_index"]
                    current["peak_bbox"] = det["bbox"]
        if current is not None:
            intervals.append(current)

    intervals.sort(key=lambda iv: (iv["start_timestamp"], iv["class_name"]))
    counters: dict[str, int] = defaultdict(int)
    for interval in intervals:
        index = counters[interval["class_name"]]
        counters[interval["class_name"]] += 1
        interval["interval_id"] = f"{interval['class_name']}_{index}"
    return intervals


def extract_crops(video_path: Path, video_name: str, intervals: list[dict]) -> None:
    cap = cv2.VideoCapture(str(video_path))
    try:
        for interval in intervals:
            cap.set(cv2.CAP_PROP_POS_FRAMES, interval["peak_frame_index"])
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    f"Cannot read frame {interval['peak_frame_index']} from: {video_path}"
                )
            x1, y1, x2, y2 = interval["peak_bbox"]
            crop = frame[y1:y2, x1:x2]
            crop_path = CROPS_ROOT / video_name / f"{interval['interval_id']}.jpg"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(crop_path), crop):
                raise RuntimeError(f"Failed to write crop: {crop_path}")
            interval["crop_path"] = str(crop_path)
    finally:
        cap.release()


def build_report(
    video_name: str,
    metadata: dict,
    intervals: list[dict],
    frame_interval: int,
) -> dict:
    sampling_window = frame_interval / metadata["fps"]
    by_class: dict[str, list[dict]] = defaultdict(list)
    for interval in intervals:
        by_class[interval["class_name"]].append(interval)

    classes = {}
    for class_name, class_intervals in by_class.items():
        entries = []
        total_visible = 0.0
        for interval in class_intervals:
            duration = max(
                interval["end_timestamp"] - interval["start_timestamp"],
                sampling_window,
            )
            total_visible += duration
            entries.append(
                {
                    "interval_id": interval["interval_id"],
                    "start_time": interval["start_timestamp"],
                    "end_time": interval["end_timestamp"],
                    "duration": duration,
                    "peak_confidence": interval["peak_confidence"],
                    "crop_path": interval["crop_path"],
                }
            )
        classes[class_name] = {
            "total_visible_seconds": round(total_visible, 3),
            "percentage": round(100 * total_visible / metadata["duration_seconds"], 2),
            "intervals": entries,
        }

    return {
        "video_name": video_name,
        "fps": metadata["fps"],
        "total_frames": metadata["total_frames"],
        "duration_seconds": metadata["duration_seconds"],
        "classes": classes,
    }


def load_model() -> YOLO:
    weights = hf_hub_download(repo_id=WEIGHTS_REPO, filename=WEIGHTS_FILENAME)
    return YOLO(weights)


def process_video(
    source: str,
    frame_interval: int = DEFAULT_FRAME_INTERVAL,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    gap_tolerance: int = DEFAULT_GAP_TOLERANCE,
) -> dict:
    video_path = resolve_video_path(source)
    video_name = re.sub(r"[^\w\-]", "_", video_path.stem)
    metadata = read_video_metadata(video_path)
    model = load_model()
    detections = detect_frames(
        model, video_path, metadata, frame_interval, conf_threshold
    )
    intervals = group_intervals(detections, gap_tolerance)
    extract_crops(video_path, video_name, intervals)
    return build_report(video_name, metadata, intervals, frame_interval)


def stream_video(
    source: str,
    frame_interval: int = DEFAULT_FRAME_INTERVAL,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    gap_tolerance: int = DEFAULT_GAP_TOLERANCE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Generator[dict]:
    video_path = resolve_video_path(source)
    video_name = re.sub(r"[^\w\-]", "_", video_path.stem)
    metadata = read_video_metadata(video_path)
    model = load_model()
    fps = metadata["fps"]
    total_frames = metadata["total_frames"]

    open_intervals: dict[str, dict] = {}
    counters: dict[str, int] = defaultdict(int)
    closed_intervals: list[dict] = []

    def apply_batch(batch_frames: list, batch_indices: list[int]) -> None:
        results = model.predict(batch_frames, conf=conf_threshold, verbose=False)
        for frame_index, frame, result in zip(batch_indices, batch_frames, results):
            timestamp = frame_index / fps
            for box in result.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                class_name = model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                interval = open_intervals.get(class_name)
                if interval is None:
                    open_intervals[class_name] = {
                        "class_name": class_name,
                        "first_frame_index": frame_index,
                        "last_frame_index": frame_index,
                        "start_timestamp": timestamp,
                        "end_timestamp": timestamp,
                        "peak_confidence": confidence,
                        "peak_frame_index": frame_index,
                        "peak_bbox": (x1, y1, x2, y2),
                        "peak_frame_crop": frame,
                    }
                else:
                    interval["last_frame_index"] = frame_index
                    interval["end_timestamp"] = timestamp
                    if confidence > interval["peak_confidence"]:
                        interval["peak_confidence"] = confidence
                        interval["peak_frame_index"] = frame_index
                        interval["peak_bbox"] = (x1, y1, x2, y2)
                        interval["peak_frame_crop"] = frame

    def close_interval(class_name: str) -> dict:
        interval = open_intervals.pop(class_name)
        interval_id = f"{class_name}_{counters[class_name]}"
        counters[class_name] += 1
        x1, y1, x2, y2 = interval["peak_bbox"]
        crop = interval["peak_frame_crop"][y1:y2, x1:x2]
        crop_path = CROPS_ROOT / video_name / f"{interval_id}.jpg"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(crop_path), crop):
            raise RuntimeError(f"Failed to write crop: {crop_path}")
        interval["interval_id"] = interval_id
        interval["crop_path"] = str(crop_path)
        closed_intervals.append(interval)
        return {
            "interval_id": interval_id,
            "start_time": interval["start_timestamp"],
            "end_time": interval["end_timestamp"],
            "duration": max(
                interval["end_timestamp"] - interval["start_timestamp"],
                frame_interval / fps,
            ),
            "peak_confidence": interval["peak_confidence"],
            "crop_path": str(crop_path),
        }

    def close_expired(last_frame_index: int) -> Generator[dict]:
        for class_name in list(open_intervals):
            if (
                last_frame_index - open_intervals[class_name]["last_frame_index"]
                > gap_tolerance
            ):
                yield {"type": "interval", "data": close_interval(class_name)}

    cap = cv2.VideoCapture(str(video_path))
    try:
        batch_frames: list = []
        batch_indices: list[int] = []
        for frame_index in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            batch_frames.append(frame)
            batch_indices.append(frame_index)
            if len(batch_frames) < batch_size:
                continue
            apply_batch(batch_frames, batch_indices)
            yield from close_expired(batch_indices[-1])
            batch_frames = []
            batch_indices = []
        if batch_frames:
            apply_batch(batch_frames, batch_indices)
    finally:
        cap.release()

    for class_name in list(open_intervals):
        yield {"type": "interval", "data": close_interval(class_name)}

    closed_intervals.sort(key=lambda iv: (iv["start_timestamp"], iv["class_name"]))
    yield {
        "type": "report",
        "data": build_report(video_name, metadata, closed_intervals, frame_interval),
    }


def write_to_supabase(report: dict) -> None:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise OSError("Set SUPABASE_URL and SUPABASE_KEY environment variables")

    client = create_client(url, key)

    client.table("videos").insert(
        {
            "video_name": report["video_name"],
            "duration_seconds": report["duration_seconds"],
            "fps": report["fps"],
            "total_frames": report["total_frames"],
            "processed_at": datetime.now(UTC).isoformat(),
        }
    ).execute()

    detection_rows = []
    uploads = []
    for class_name, class_report in report["classes"].items():
        for interval in class_report["intervals"]:
            storage_file = f"{report['video_name']}/{interval['interval_id']}.jpg"
            detection_rows.append(
                {
                    "video_name": report["video_name"],
                    "brand": class_name,
                    "start_time": interval["start_time"],
                    "end_time": interval["end_time"],
                    "duration": interval["duration"],
                    "peak_confidence": interval["peak_confidence"],
                    "crop_path": f"{STORAGE_BUCKET}/{storage_file}",
                }
            )
            uploads.append((interval["crop_path"], storage_file))

    if detection_rows:
        client.table("detections").insert(detection_rows).execute()

    for local_path, storage_file in uploads:
        with open(local_path, "rb") as crop:
            client.storage.from_(STORAGE_BUCKET).upload(
                storage_file,
                crop,
                {"content-type": "image/jpeg", "x-upsert": "true"},
            )
