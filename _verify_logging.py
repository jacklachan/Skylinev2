"""
Headless verification: opens webcam, runs YOLOv8 + YOLO-World inference for
N frames using the same code path as _camera_loop, writes CSVs via metrics,
then prints a summary. No GUI required.
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cv2
from ultralytics import YOLO, YOLOWorld
from PIL import Image
import metrics
from app import (
    MODEL_PATH, EXTRA_MODEL_PATH, EXTRA_CLASSES,
    EXTRA_CONFIDENCE, EXTRA_IOU, DISPLAY_SIZE, EXTRA_CLASS_ALIASES,
)

N_FRAMES = 20


def normalize_extra_label(label, aliases=EXTRA_CLASS_ALIASES):
    normalized = label.lower().strip()
    for cls_name, al in aliases.items():
        if normalized in al:
            return cls_name
    return None


def draw_extra_detections(frame, result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return frame, {"tree": 0, "building": 0}
    names = result.names
    counts = {"tree": 0, "building": 0}
    for box in boxes:
        cls_id = int(box.cls[0])
        label = normalize_extra_label(names.get(cls_id, ""))
        if label:
            counts[label] += 1
    return frame, counts


def main():
    print("Loading models...")
    model = YOLO(MODEL_PATH)
    try:
        extra_model = YOLOWorld(EXTRA_MODEL_PATH)
        extra_model.set_classes(EXTRA_CLASSES)
        print(f"  YOLOv8:     {MODEL_PATH}")
        print(f"  YOLO-World: {EXTRA_MODEL_PATH}")
    except Exception as e:
        extra_model = None
        print(f"  YOLO-World unavailable: {e}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam (index 0)")
        sys.exit(1)
    print(f"\nWebcam opened. Running {N_FRAMES} frames...\n")

    metrics.init_csv_files()

    for idx in range(N_FRAMES):
        t0 = time.perf_counter()
        ret, frame = cap.read()
        capture_ms = (time.perf_counter() - t0) * 1000
        if not ret:
            print(f"Frame {idx}: read failed, skipping")
            continue

        ts = time.strftime("%Y-%m-%dT%H:%M:%S")

        t_yolo8 = time.perf_counter()
        results = model(frame, verbose=False)
        yolov8_ms = (time.perf_counter() - t_yolo8) * 1000

        t_world = time.perf_counter()
        extra_results = None
        if extra_model is not None:
            extra_results = extra_model(frame, verbose=False, conf=EXTRA_CONFIDENCE, iou=EXTRA_IOU)
        yoloworld_ms = (time.perf_counter() - t_world) * 1000 if extra_results is not None else 0.0

        t_ann = time.perf_counter()
        annotated = results[0].plot()
        if extra_results is not None:
            annotated, _ = draw_extra_detections(annotated, extra_results[0])
        annotated = cv2.resize(annotated, DISPLAY_SIZE)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        Image.fromarray(annotated)
        annotate_ms = (time.perf_counter() - t_ann) * 1000

        metrics.log_latency(idx, ts, capture_ms, yolov8_ms, yoloworld_ms, annotate_ms)
        metrics.log_detections(ts, "yolov8", results[0])
        if extra_results is not None:
            metrics.log_detections(ts, "yoloworld", extra_results[0])

        total = capture_ms + yolov8_ms + yoloworld_ms + annotate_ms
        n_det = len(results[0].boxes) if results[0].boxes is not None else 0
        print(f"  frame {idx:>3}  capture={capture_ms:6.1f}ms  yolov8={yolov8_ms:6.1f}ms  "
              f"world={yoloworld_ms:6.1f}ms  ann={annotate_ms:5.1f}ms  "
              f"total={total:6.1f}ms  dets={n_det}")

    cap.release()
    print("\nDone. Checking CSVs...")

    for path in [metrics.LATENCY_CSV, metrics.DETECTIONS_CSV]:
        with open(path) as f:
            lines = f.readlines()
        print(f"  {path}: {len(lines)-1} data rows")
        if len(lines) > 1:
            print(f"    header: {lines[0].strip()}")
            print(f"    last:   {lines[-1].strip()}")

    print("\nRun  python aggregate.py  to see aggregated stats.")


if __name__ == "__main__":
    main()
