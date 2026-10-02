"""
Object Detection + Monocular Depth Estimation + Distance Measurement
======================================================================
Detects objects in a video with YOLOv8, estimates per-pixel depth with
Depth Anything V2, and overlays:
  - a bounding box + class name + confidence + depth + approx. distance (m)
    for every detected object
  - a colorized depth heatmap in the corner
  - dashed red lines + distance labels between every pair of detected people

Outputs:
  - VIDEO_OUT_PATH: annotated video
  - CSV_OUT_PATH:   per-object detections with depth + approx. distance
"""

import csv
import math

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from transformers import pipeline

# ============================== CONFIG ==============================

VIDEO_IN_PATH = "input.mp4"
VIDEO_OUT_PATH = "output_with_depth.mp4"
CSV_OUT_PATH = "detections_with_depth.csv"

YOLO_WEIGHTS = "yolov8n.pt"
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

IMG_SIZE_YOLO = 640          # YOLO inference resolution (speed vs. accuracy)
DEVICE = "cpu"               # "cpu" on Mac (no CUDA); use 0 for an NVIDIA GPU
CONF_THRESH = 0.25           # minimum YOLO confidence to keep a detection
BOX_WINDOW = 5                # px window used to sample median depth in a box
SKIP_EVERY_N = 1              # process every Nth frame (1 = every frame)

HEATMAP_WIDTH_FRACTION = 0.3   # depth heatmap corner overlay size (30% of frame width)

# Assumed horizontal field of view of the recording camera, in degrees.
# Typical phone cameras are ~65-78 degrees. Adjust for better accuracy
# if you know your exact device's horizontal FOV.
ASSUMED_HFOV_DEG = 70.0

# Real-world average heights (meters), used to estimate metric distance
# from how tall an object appears in the frame (its pixel height).
KNOWN_HEIGHTS_M = {
    "person": 1.7,
    "car": 1.5,
}

# =====================================================================


# ----------------------------- Geometry -----------------------------

def estimate_metric_distance(class_name, pixel_height, focal_length_px):
    """Estimate distance from camera (meters) using a known real-world height.

    distance = (real_height_m * focal_length_px) / pixel_height

    Returns None for classes without a known reference height.
    """
    real_height_m = KNOWN_HEIGHTS_M.get(class_name)
    if real_height_m is None or pixel_height <= 0:
        return None
    return (real_height_m * focal_length_px) / pixel_height


def angle_from_camera_axis(center_x, frame_width, focal_length_px):
    """Horizontal angle (radians) of a point from the camera's optical axis."""
    return math.atan2(center_x - frame_width / 2.0, focal_length_px)


def law_of_cosines_distance(z1, theta1, z2, theta2):
    """Distance between two points at depths z1, z2 separated by angle (theta1 - theta2)."""
    delta = theta1 - theta2
    squared = z1 ** 2 + z2 ** 2 - 2 * z1 * z2 * math.cos(delta)
    return math.sqrt(max(squared, 0.0))


def bbox_median_depth(depth_map, x1, y1, x2, y2, window=BOX_WINDOW):
    """Median depth in a small central window of a bounding box (robust to edge noise)."""
    h, w = depth_map.shape
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(w - 1, int(round(x2))), min(h - 1, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return None

    if window > 1:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half = window // 2
        wx1, wy1 = max(0, cx - half), max(0, cy - half)
        wx2, wy2 = min(w - 1, cx + half), min(h - 1, cy + half)
        central = depth_map[wy1:wy2 + 1, wx1:wx2 + 1]
        if central.size > 0:
            return float(np.median(central))

    roi = depth_map[y1:y2 + 1, x1:x2 + 1]
    return float(np.median(roi)) if roi.size > 0 else None


# ------------------------------ Drawing ------------------------------

def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=12):
    """Draws a dashed line (OpenCV has no built-in dashed line primitive)."""
    (x1, y1), (x2, y2) = pt1, pt2
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    n_dashes = max(1, int(length / dash_length))
    for i in range(n_dashes):
        start_frac = i / n_dashes
        end_frac = (i + 0.5) / n_dashes  # draw half, skip half -> dash pattern
        start = (int(x1 + (x2 - x1) * start_frac), int(y1 + (y2 - y1) * start_frac))
        end = (int(x1 + (x2 - x1) * end_frac), int(y1 + (y2 - y1) * end_frac))
        cv2.line(img, start, end, color, thickness)


def draw_label(img, text, anchor_xy, bg_color, text_color=(0, 0, 0), font_scale=0.5, thickness=1):
    """Draws text with a filled background box above/at the given anchor point."""
    x, y = anchor_xy
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (x, y - th - 6), (x + tw, y), bg_color, -1)
    cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)


def draw_detection_box(img, detection):
    """Draws a bounding box + label for a single detection dict."""
    x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
    color = (0, 255, 0)
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    dist_txt = f", ~{detection['dist_m']:.1f}m" if detection["dist_m"] is not None else ""
    depth_txt = f"{detection['depth']:.3f}" if detection["depth"] is not None else "N/A"
    label = f"cls:{detection['class_id']}, {detection['class_name']}, conf:{detection['conf']:.2f} d:{depth_txt}{dist_txt}"
    draw_label(img, label, (int(x1), int(y1)), bg_color=color)


def draw_person_links(img, people):
    """Draws a dashed red line + distance label between every pair of detected people."""
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            a, b = people[i], people[j]
            dist = law_of_cosines_distance(a["depth_m"], a["theta"], b["depth_m"], b["theta"])
            draw_dashed_line(img, a["center"], b["center"], color=(0, 0, 255))

            mid = (int((a["center"][0] + b["center"][0]) / 2), int((a["center"][1] + b["center"][1]) / 2))
            draw_label(img, f"~{dist:.1f}m", mid, bg_color=(0, 0, 255), text_color=(255, 255, 255),
                       font_scale=0.6, thickness=2)


def build_depth_heatmap(depth_map):
    """Converts a raw depth map into a colorized (JET) heatmap image."""
    normalized = depth_map.astype(np.float32)
    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)
    normalized = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def overlay_heatmap_corner(frame, heatmap, width_fraction=HEATMAP_WIDTH_FRACTION):
    """Pastes a resized heatmap into the top-left corner of the frame, in place."""
    frame_h, frame_w = frame.shape[:2]
    heat_h, heat_w = heatmap.shape[:2]

    thumb_w = int(frame_w * width_fraction)
    thumb_h = int(thumb_w * (heat_h / heat_w))
    thumbnail = cv2.resize(heatmap, (thumb_w, thumb_h))
    frame[0:thumb_h, 0:thumb_w] = thumbnail


# --------------------------- Model / IO setup ---------------------------

def load_models():
    print("Loading YOLO model...")
    yolo_model = YOLO(YOLO_WEIGHTS)

    print("Loading Depth Anything V2 (Hugging Face) pipeline...")
    depth_model = pipeline(task="depth-estimation", model=DEPTH_MODEL_ID, device=DEVICE)

    return yolo_model, depth_model


def open_input_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    }
    return cap, info


def open_output_video(path, fps, width, height):
    # 'avc1' = H.264. Far more reliable playback on macOS/QuickTime than the
    # legacy 'mp4v' (MPEG-4 Simple Profile) codec, which can appear blocky
    # or "break up" during playback with a lot of motion/overlays.
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def estimate_focal_length_px(frame_width, hfov_deg=ASSUMED_HFOV_DEG):
    hfov_rad = math.radians(hfov_deg)
    return (frame_width / 2.0) / math.tan(hfov_rad / 2.0)


# ------------------------------ Per-frame ------------------------------

def run_yolo(yolo_model, frame):
    """Runs YOLO on one frame; returns (boxes, scores, classes, names_dict)."""
    results = yolo_model.predict(source=frame, imgsz=IMG_SIZE_YOLO, conf=CONF_THRESH, verbose=False)
    result = results[0]

    if not hasattr(result.boxes, "xyxy"):
        return np.array([]), [], [], result.names

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    return boxes, scores, classes, result.names


def run_depth(depth_model, frame_bgr, target_width, target_height):
    """Runs the depth model on one frame; returns a depth map resized to the frame size."""
    pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    output = depth_model(pil_image)
    if isinstance(output, list):
        output = output[0]
    depth_map = output["depth"] if isinstance(output, dict) and "depth" in output else output

    depth_map = np.asarray(depth_map)
    if depth_map.ndim == 3 and depth_map.shape[2] == 1:
        depth_map = depth_map[:, :, 0]

    if depth_map.shape[:2] != (target_height, target_width):
        depth_map = cv2.resize(depth_map, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return depth_map


def build_detections(boxes, scores, classes, class_names, depth_map, focal_length_px, frame_width):
    """Combines YOLO + depth outputs into a list of detection dicts, ready to draw/log."""
    detections = []
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        cls_id = int(cls)
        cls_name = class_names.get(cls_id, "unknown")

        depth = bbox_median_depth(depth_map, x1, y1, x2, y2)
        pixel_height = y2 - y1
        dist_m = estimate_metric_distance(cls_name, pixel_height, focal_length_px)

        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        theta = angle_from_camera_axis(center[0], frame_width, focal_length_px)

        detections.append({
            "class_id": cls_id, "class_name": cls_name, "conf": float(score),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "depth": depth, "dist_m": dist_m,
            "center": center, "theta": theta,
        })
    return detections


def write_detections_csv(csv_writer, frame_idx, detections):
    for det_idx, det in enumerate(detections):
        csv_writer.writerow([
            frame_idx, det_idx, det["class_id"], det["class_name"], det["conf"],
            float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"]),
            det["depth"], det["dist_m"],
        ])


def render_frame(frame, detections, depth_map):
    """Draws boxes/labels, person-to-person links, and the depth heatmap onto a copy of the frame."""
    out_frame = frame.copy()

    for det in detections:
        draw_detection_box(out_frame, det)

    people = [
        {"center": d["center"], "depth_m": d["dist_m"], "theta": d["theta"]}
        for d in detections if d["class_name"] == "person" and d["dist_m"] is not None
    ]
    draw_person_links(out_frame, people)

    heatmap = build_depth_heatmap(depth_map)
    overlay_heatmap_corner(out_frame, heatmap)

    return out_frame


# --------------------------------- Main ---------------------------------

def main():
    yolo_model, depth_model = load_models()
    cap, video_info = open_input_video(VIDEO_IN_PATH)
    width, height, fps, total_frames = (
        video_info["width"], video_info["height"], video_info["fps"], video_info["total_frames"]
    )
    print(f"Video opened: {VIDEO_IN_PATH} ({width}x{height}) @ {fps} FPS, {total_frames} frames")

    focal_length_px = estimate_focal_length_px(width)
    print(f"Assumed horizontal FOV: {ASSUMED_HFOV_DEG} degrees -> estimated focal length = {focal_length_px:.2f} px")

    out_video = open_output_video(VIDEO_OUT_PATH, fps / max(1, SKIP_EVERY_N), width, height)

    csv_file = open(CSV_OUT_PATH, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_idx", "det_idx", "class_id", "class_name", "conf",
        "x1", "y1", "x2", "y2", "median_depth", "approx_distance_m",
    ])

    frame_idx = 0
    processed_frames = 0
    progress = tqdm(total=total_frames, desc="Frames")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            progress.update(1)

            if frame_idx % SKIP_EVERY_N != 0:
                frame_idx += 1
                continue

            boxes, scores, classes, class_names = run_yolo(yolo_model, frame)
            depth_map = run_depth(depth_model, frame, width, height)
            detections = build_detections(boxes, scores, classes, class_names, depth_map, focal_length_px, width)

            write_detections_csv(csv_writer, frame_idx, detections)
            out_frame = render_frame(frame, detections, depth_map)
            out_video.write(out_frame)

            processed_frames += 1
            frame_idx += 1
    finally:
        progress.close()
        cap.release()
        out_video.release()
        csv_file.close()
        print(f"Done. Processed frames: {processed_frames}. Output video: {VIDEO_OUT_PATH}, CSV: {CSV_OUT_PATH}")


if __name__ == "__main__":
    main()