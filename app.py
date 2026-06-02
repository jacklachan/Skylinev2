import os
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|max_delay;0|fflags;nobuffer|flags;low_delay",
)

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO, YOLOWorld

import metrics

LOGGING = True

MODEL_PATH = "yolov8n.pt"
EXTRA_MODEL_PATH = "yolov8s-worldv2.pt"
EXTRA_CLASS_ALIASES = {
    "tree": ["tree", "large tree", "green tree"],
    "building": ["building", "tall building", "office building"],
}
EXTRA_CLASSES = [
    alias
    for aliases in EXTRA_CLASS_ALIASES.values()
    for alias in aliases
]
EXTRA_CONFIDENCE = 0.08
EXTRA_IOU = 0.45
DISPLAY_SIZE = (900, 560)


class YoloDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Object Detection Dashboard")
        self.root.geometry("1120x760")
        self.root.minsize(940, 640)
        self.root.configure(bg="#eef4ef")

        self.model = None
        self.extra_model = None
        self.extra_model_error = None
        self.cap = None
        self.running = False
        self.worker = None
        self.capture_worker = None
        self.cap_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_id = 0
        self.read_failures = 0
        self.frame_count = 0
        self.extra_frame_count = 0
        self.last_fps_time = time.time()
        self.photo = None
        self.latest_capture_ms = 0.0
        self._log_frame_idx = 0

        if LOGGING:
            metrics.init_csv_files()

        self._build_ui()
        self._load_model()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#153b2d", height=92)
        header.pack(fill="x")
        header.pack_propagate(False)

        title_box = tk.Frame(header, bg="#153b2d")
        title_box.pack(side="left", padx=28, pady=18)

        tk.Label(
            title_box,
            text="YOLO Object Detection",
            bg="#153b2d",
            fg="white",
            font=("Segoe UI", 24, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_box,
            text="Tkinter RTSP IP camera dashboard with live start and stop control",
            bg="#153b2d",
            fg="#cfe2d7",
            font=("Segoe UI", 11),
        ).pack(anchor="w", pady=(4, 0))

        main = tk.Frame(self.root, bg="#eef4ef")
        main.pack(fill="both", expand=True, padx=22, pady=22)

        video_panel = tk.Frame(main, bg="white", highlightthickness=1, highlightbackground="#d4dfd8")
        video_panel.pack(side="left", fill="both", expand=True)

        self.video_label = tk.Label(
            video_panel,
            text="Camera preview will appear here",
            bg="#101916",
            fg="#d7e4db",
            font=("Segoe UI", 16, "bold"),
            compound="center",
        )
        self.video_label.pack(fill="both", expand=True, padx=14, pady=14)

        side = tk.Frame(main, bg="#eef4ef", width=270)
        side.pack(side="right", fill="y", padx=(18, 0))
        side.pack_propagate(False)

        control_card = self._card(side, "Camera Control")

        tk.Label(
            control_card,
            text="RTSP URL",
            bg="white",
            fg="#1a241f",
            anchor="w",
            font=("Segoe UI", 10, "bold"),
        ).pack(fill="x", pady=(8, 4))

        self.rtsp_url_var = tk.StringVar()
        self.rtsp_url_entry = ttk.Entry(control_card, textvariable=self.rtsp_url_var)
        self.rtsp_url_entry.pack(fill="x", pady=(0, 4), ipady=5)

        tk.Label(
            control_card,
            text="Example: rtsp://user:pass@192.168.1.10:554/stream",
            bg="white",
            fg="#62716a",
            justify="left",
            anchor="w",
            font=("Segoe UI", 8),
            wraplength=220,
        ).pack(fill="x", pady=(0, 10))

        self.start_btn = ttk.Button(control_card, text="Start RTSP Camera", command=self.start_camera)
        self.start_btn.pack(fill="x", pady=(8, 4), ipady=8)

        self.webcam_btn = ttk.Button(control_card, text="Start Webcam", command=self.start_webcam)
        self.webcam_btn.pack(fill="x", pady=(0, 8), ipady=8)

        self.stop_btn = ttk.Button(control_card, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_btn.pack(fill="x", pady=(0, 8), ipady=8)

        self.snapshot_btn = ttk.Button(control_card, text="Save Snapshot", command=self.save_snapshot, state="disabled")
        self.snapshot_btn.pack(fill="x", ipady=8)

        status_card = self._card(side, "Live Status")
        self.status_var = tk.StringVar(value="Loading model...")
        self.fps_var = tk.StringVar(value="FPS: --")
        self.model_var = tk.StringVar(value=f"Model: {MODEL_PATH}")
        self.extra_model_var = tk.StringVar(value="Extra: loading tree/building...")
        self.extra_hits_var = tk.StringVar(value="Extra hits: --")
        self.camera_var = tk.StringVar(value="Source: RTSP not connected")

        for variable in (
            self.status_var,
            self.fps_var,
            self.model_var,
            self.extra_model_var,
            self.extra_hits_var,
            self.camera_var,
        ):
            tk.Label(
                status_card,
                textvariable=variable,
                bg="white",
                fg="#1a241f",
                anchor="w",
                font=("Segoe UI", 10, "bold"),
            ).pack(fill="x", pady=5)

        help_card = self._card(side, "Usage")
        tk.Label(
            help_card,
            text=(
                "1. Enter the RTSP URL of the IP camera.\n"
                "2. Press Start RTSP Camera.\n"
                "3. Objects are detected live.\n"
                "4. Press Stop Camera to release stream.\n"
                "5. Close window safely anytime."
            ),
            bg="white",
            fg="#62716a",
            justify="left",
            anchor="nw",
            font=("Segoe UI", 10),
            wraplength=220,
        ).pack(fill="both", expand=True, pady=(8, 0))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=8)

    def _card(self, parent, title):
        card = tk.Frame(parent, bg="white", highlightthickness=1, highlightbackground="#d4dfd8")
        card.pack(fill="x", pady=(0, 16))

        tk.Label(
            card,
            text=title,
            bg="white",
            fg="#1a241f",
            anchor="w",
            font=("Segoe UI", 13, "bold"),
        ).pack(fill="x", padx=16, pady=(14, 4))

        body = tk.Frame(card, bg="white")
        body.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        return body

    def _load_model(self):
        try:
            self.model = YOLO(MODEL_PATH)
            self.status_var.set("Ready")
        except Exception as exc:
            self.status_var.set("Model load failed")
            messagebox.showerror("Model Error", f"Could not load {MODEL_PATH}\n\n{exc}")
            return

        try:
            self.extra_model = YOLOWorld(EXTRA_MODEL_PATH)
            self.extra_model.set_classes(EXTRA_CLASSES)
            self.extra_model_var.set("Extra: tree/building enabled")
        except Exception as exc:
            self.extra_model = None
            self.extra_model_error = str(exc)
            self.extra_model_var.set("Extra: tree/building unavailable")
            self.extra_hits_var.set(f"Extra error: {str(exc)[:80]}")

    def start_camera(self):
        if self.running:
            return

        if self.model is None:
            messagebox.showwarning("Model Not Ready", "YOLO model is not loaded.")
            return

        rtsp_url = self.rtsp_url_var.get().strip()
        if not rtsp_url:
            messagebox.showwarning("RTSP URL Required", "Enter the RTSP URL for the IP camera.")
            return

        if not rtsp_url.lower().startswith("rtsp://"):
            messagebox.showwarning("Invalid RTSP URL", "Enter a valid RTSP URL beginning with rtsp://")
            return

        self.status_var.set("Connecting to camera...")
        self.start_btn.configure(state="disabled")
        self.webcam_btn.configure(state="disabled")
        self.rtsp_url_entry.configure(state="disabled")

        threading.Thread(target=self._connect_rtsp, args=(rtsp_url,), daemon=True).start()

    def _connect_rtsp(self, rtsp_url):
        cap = self._open_rtsp_camera(rtsp_url)
        if not cap.isOpened():
            cap.release()
            self.root.after(0, self._on_rtsp_failed)
            return
        self.root.after(0, self._on_rtsp_connected, cap)

    def _on_rtsp_failed(self):
        self.status_var.set("Ready")
        self.start_btn.configure(state="normal")
        self.webcam_btn.configure(state="normal")
        self.rtsp_url_entry.configure(state="normal")
        messagebox.showerror("Camera Error", "Could not open RTSP camera. Check the URL, network, username, and password.")

    def _on_rtsp_connected(self, cap):
        self.cap = cap
        self.running = True
        self.latest_frame = None
        self.latest_frame_id = 0
        self.read_failures = 0
        self.frame_count = 0
        self.extra_frame_count = 0
        self.last_fps_time = time.time()
        self.status_var.set("RTSP camera running")
        self.camera_var.set("Source: RTSP stream")
        self.stop_btn.configure(state="normal")
        self.snapshot_btn.configure(state="normal")

        self.capture_worker = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_worker.start()

        self.worker = threading.Thread(target=self._camera_loop, daemon=True)
        self.worker.start()

    def start_webcam(self):
        if self.running:
            return
        if self.model is None:
            messagebox.showwarning("Model Not Ready", "YOLO model is not loaded.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            messagebox.showerror("Webcam Error", "Could not open webcam (index 0).")
            return

        self.cap = cap
        self.running = True
        self.latest_frame = None
        self.latest_frame_id = 0
        self.read_failures = 0
        self.frame_count = 0
        self.extra_frame_count = 0
        self.last_fps_time = time.time()
        self.status_var.set("Webcam running")
        self.camera_var.set("Source: Webcam (0)")
        self.rtsp_url_entry.configure(state="disabled")
        self.start_btn.configure(state="disabled")
        self.webcam_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.snapshot_btn.configure(state="normal")

        self.capture_worker = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_worker.start()

        self.worker = threading.Thread(target=self._camera_loop, daemon=True)
        self.worker.start()

    def _open_rtsp_camera(self, rtsp_url):
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def stop_camera(self):
        self.running = False
        self.status_var.set("Stopping camera...")
        self.rtsp_url_entry.configure(state="normal")
        self.start_btn.configure(state="normal")
        self.webcam_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.snapshot_btn.configure(state="disabled")

        self.video_label.configure(image="", text="Camera stopped")
        self.status_var.set("Stopped")
        self.camera_var.set("Source: RTSP not connected")
        self.fps_var.set("FPS: --")

    def _capture_loop(self):
        while self.running:
            with self.cap_lock:
                cap = self.cap

            if cap is None:
                break

            t0 = time.perf_counter()
            ret, frame = cap.read()
            capture_ms = (time.perf_counter() - t0) * 1000
            if not self.running:
                break

            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_frame_id += 1
                    self.latest_capture_ms = capture_ms
                self.read_failures = 0
                continue

            self.read_failures += 1
            if self.read_failures >= 15:
                self.root.after(0, self._handle_camera_error)
                break
            time.sleep(0.03)

        with self.cap_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def _camera_loop(self):
        last_processed_id = -1

        while self.running:
            with self.frame_lock:
                frame_id = self.latest_frame_id
                frame = None if self.latest_frame is None else self.latest_frame.copy()
                capture_ms = self.latest_capture_ms

            if frame is None or frame_id == last_processed_id:
                time.sleep(0.005)
                continue

            last_processed_id = frame_id
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")

            t_yolo8 = time.perf_counter()
            results = self.model(frame, verbose=False)
            t_yolo8_end = time.perf_counter()

            t_world = time.perf_counter()
            extra_results = None
            if self.extra_model is not None:
                extra_results = self.extra_model(
                    frame,
                    verbose=False,
                    conf=EXTRA_CONFIDENCE,
                    iou=EXTRA_IOU,
                )
            t_world_end = time.perf_counter()

            t_ann = time.perf_counter()
            annotated = results[0].plot()
            if extra_results is not None:
                annotated, extra_counts = self._draw_extra_detections(annotated, extra_results[0])
                self.extra_frame_count += 1
                if self.extra_frame_count % 5 == 0:
                    self.root.after(0, self._update_extra_hits, extra_counts)
            annotated = cv2.resize(annotated, DISPLAY_SIZE)
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(annotated)
            t_ann_end = time.perf_counter()

            if LOGGING:
                yolov8_ms = (t_yolo8_end - t_yolo8) * 1000
                yoloworld_ms = (t_world_end - t_world) * 1000 if extra_results is not None else 0.0
                annotate_ms = (t_ann_end - t_ann) * 1000
                metrics.log_latency(self._log_frame_idx, ts, capture_ms, yolov8_ms, yoloworld_ms, annotate_ms)
                metrics.log_detections(ts, "yolov8", results[0])
                if extra_results is not None:
                    metrics.log_detections(ts, "yoloworld", extra_results[0])
                self._log_frame_idx += 1

            self.frame_count += 1
            now = time.time()
            if now - self.last_fps_time >= 1:
                fps = self.frame_count / (now - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = now
                self.root.after(0, lambda value=fps: self.fps_var.set(f"FPS: {value:.1f}"))

            self.root.after(0, self._update_frame, image)

    def _draw_extra_detections(self, frame, result):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return frame, {"tree": 0, "building": 0}

        names = result.names
        counts = {"tree": 0, "building": 0}
        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            raw_label = names.get(cls_id, str(cls_id))
            label = self._normalize_extra_label(raw_label)
            if label is None:
                continue

            counts[label] += 1
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]

            color = (42, 180, 67) if label == "tree" else (255, 145, 35)
            text = f"{label} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            text_w, text_h = text_size
            label_y1 = max(y1 - text_h - baseline - 6, 0)
            cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 8, y1), color, -1)
            cv2.putText(
                frame,
                text,
                (x1 + 4, max(y1 - baseline - 4, text_h + 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return frame, counts

    def _normalize_extra_label(self, label):
        normalized = label.lower().strip()
        for class_name, aliases in EXTRA_CLASS_ALIASES.items():
            if normalized in aliases:
                return class_name
        return None

    def _update_extra_hits(self, counts):
        self.extra_hits_var.set(
            f"Extra hits: tree {counts['tree']} | building {counts['building']}"
        )

    def _update_frame(self, image):
        self.photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.photo, text="")

    def _handle_camera_error(self):
        self.stop_camera()
        messagebox.showerror("Camera Error", "RTSP camera frame could not be read.")

    def save_snapshot(self):
        if self.photo is None:
            return

        os.makedirs("snapshots", exist_ok=True)
        filename = f"snapshots/yolo_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        image = ImageTk.getimage(self.photo)
        image.save(filename)
        self.status_var.set(f"Saved {filename}")

    def close(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.root.destroy()


if __name__ == "__main__":
    app_root = tk.Tk()
    YoloDashboard(app_root)
    app_root.mainloop()
