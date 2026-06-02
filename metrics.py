import csv
import os
import threading

RESULTS_DIR = "results"
LATENCY_CSV = os.path.join(RESULTS_DIR, "latency.csv")
DETECTIONS_CSV = os.path.join(RESULTS_DIR, "detections.csv")

LATENCY_HEADERS = ["frame_idx", "timestamp", "capture_ms", "yolov8_ms", "yoloworld_ms", "annotate_ms"]
DETECTION_HEADERS = ["timestamp", "model", "class", "confidence"]

_lock = threading.Lock()


def init_csv_files():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for path, headers in [(LATENCY_CSV, LATENCY_HEADERS), (DETECTIONS_CSV, DETECTION_HEADERS)]:
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(headers)


def log_latency(frame_idx, timestamp, capture_ms, yolov8_ms, yoloworld_ms, annotate_ms):
    with _lock:
        with open(LATENCY_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                frame_idx, timestamp,
                f"{capture_ms:.3f}", f"{yolov8_ms:.3f}",
                f"{yoloworld_ms:.3f}", f"{annotate_ms:.3f}",
            ])


def log_detections(timestamp, model_name, result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return
    names = result.names
    rows = [
        [timestamp, model_name, names.get(int(box.cls[0]), str(int(box.cls[0]))), f"{float(box.conf[0]):.4f}"]
        for box in boxes
    ]
    with _lock:
        with open(DETECTIONS_CSV, "a", newline="") as f:
            csv.writer(f).writerows(rows)
