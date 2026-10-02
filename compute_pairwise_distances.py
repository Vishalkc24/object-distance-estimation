import csv
import math
from collections import defaultdict

import cv2

# ----------------- CONFIG -----------------
VIDEO_PATH = "input.mp4"                          # same video used in part 1
DETECTIONS_CSV = "detections_with_depth.csv"       # output from part 1 script
OUTPUT_CSV = "pairwise_distances.csv"              # this script's output

# Assumed horizontal field of view of the camera, in degrees.
# Most phone main/wide cameras are roughly 65-78 degrees horizontal FOV.
# If you know your exact phone model, look up its horizontal FOV and
# change this number for a more accurate result.
ASSUMED_HFOV_DEG = 70.0

# Only compute distances for detections with at least this confidence,
# to avoid noisy/uncertain boxes skewing results.
MIN_CONF = 0.25
# -------------------------------------------


def get_video_dimensions(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def load_detections(csv_path):
    """Group detections by frame_idx. Each detection keeps class name, midpoint, depth."""
    frames = defaultdict(list)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                conf = float(row["conf"])
            except (ValueError, TypeError):
                continue
            if conf < MIN_CONF:
                continue
            try:
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
                depth = float(row["median_depth"])
            except (ValueError, TypeError):
                continue

            frame_idx = int(row["frame_idx"])
            u = (x1 + x2) / 2.0  # horizontal midpoint (pixels)
            v = (y1 + y2) / 2.0  # vertical midpoint (pixels), unused in this simplified 2D model

            frames[frame_idx].append({
                "class_name": row.get("class_name", "unknown"),
                "class_id": row.get("class_id", ""),
                "det_idx": row.get("det_idx", ""),
                "u": u,
                "v": v,
                "depth": depth,
                "conf": conf,
            })
    return frames


def compute_distance(z1, theta1, z2, theta2):
    """Law of cosines: distance between two points at depths z1, z2 separated by angle delta."""
    delta = theta1 - theta2
    value = z1 ** 2 + z2 ** 2 - 2 * z1 * z2 * math.cos(delta)
    value = max(value, 0.0)  # guard against tiny negative values from floating point error
    return math.sqrt(value)


def main():
    width, height = get_video_dimensions(VIDEO_PATH)
    cx = width / 2.0

    # Estimate focal length in pixels from the assumed horizontal FOV
    hfov_rad = math.radians(ASSUMED_HFOV_DEG)
    fx = cx / math.tan(hfov_rad / 2.0)

    print(f"Video size: {width}x{height}")
    print(f"Assumed horizontal FOV: {ASSUMED_HFOV_DEG} degrees -> estimated fx = {fx:.2f} px")

    frames = load_detections(DETECTIONS_CSV)
    print(f"Loaded detections for {len(frames)} frames from {DETECTIONS_CSV}")

    rows_written = 0
    with open(OUTPUT_CSV, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow([
            "frame_idx",
            "obj1_class", "obj1_det_idx", "obj1_conf",
            "obj2_class", "obj2_det_idx", "obj2_conf",
            "estimated_distance",
        ])

        for frame_idx, detections in sorted(frames.items()):
            n = len(detections)
            if n < 2:
                continue  # need at least 2 objects to compute a distance

            # compute angle (from camera's forward axis) for every detection in this frame
            for d in detections:
                d["theta"] = math.atan2(d["u"] - cx, fx)

            # every unique pair in this frame
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = detections[i], detections[j]
                    dist = compute_distance(a["depth"], a["theta"], b["depth"], b["theta"])
                    writer.writerow([
                        frame_idx,
                        a["class_name"], a["det_idx"], a["conf"],
                        b["class_name"], b["det_idx"], b["conf"],
                        round(dist, 3),
                    ])
                    rows_written += 1

    print(f"Done. Wrote {rows_written} pairwise distance rows to {OUTPUT_CSV}")
    print("\nNOTE: depth values from Depth-Anything-V2-Small are RELATIVE, not metric.")
    print("These distances are proportional/relative estimates, not real-world meters,")
    print("unless you later switch to a metric depth checkpoint.")


if __name__ == "__main__":
    main()
