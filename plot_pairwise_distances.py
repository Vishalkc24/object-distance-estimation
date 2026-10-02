"""Plot pairwise object distances against video frame or elapsed time."""

import argparse
import csv
from collections import defaultdict
from statistics import median

import matplotlib.pyplot as plt


DEFAULT_INPUT_CSV = "pairwise_distances.csv"
DEFAULT_OUTPUT_IMAGE = "pairwise_distances_plot.png"


def load_rows(csv_path, object_class=None):
    """Load valid pairwise-distance rows, optionally keeping one class pair."""
    rows = []
    with open(csv_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                frame_idx = int(row["frame_idx"])
                distance = float(row["estimated_distance"])
            except (KeyError, TypeError, ValueError):
                continue

            class_a = row.get("obj1_class", "unknown")
            class_b = row.get("obj2_class", "unknown")
            if object_class and (class_a != object_class or class_b != object_class):
                continue

            pair_label = (
                f"{class_a} #{row.get('obj1_det_idx', '?')} - "
                f"{class_b} #{row.get('obj2_det_idx', '?')}"
            )
            rows.append({
                "frame_idx": frame_idx,
                "distance": distance,
                "pair_label": pair_label,
            })
    return rows


def plot_rows(rows, output_path, fps=None, title=None, show_pairs=False):
    """Plot pairwise observations and save a labeled chart."""
    if not rows:
        raise ValueError("No valid pairwise-distance rows matched the requested filter")

    use_time_axis = fps is not None and fps > 0

    if show_pairs:
        grouped_rows = defaultdict(list)
        for row in rows:
            grouped_rows[row["pair_label"]].append(row)

        for pair_label, pair_rows in sorted(grouped_rows.items()):
            pair_rows.sort(key=lambda row: row["frame_idx"])
            x_values = [row["frame_idx"] / fps for row in pair_rows] if use_time_axis else [
                row["frame_idx"] for row in pair_rows
            ]
            y_values = [row["distance"] for row in pair_rows]
            plt.plot(x_values, y_values, marker=".", linewidth=1.2, markersize=4, label=pair_label)
    else:
        grouped_frames = defaultdict(list)
        for row in rows:
            grouped_frames[row["frame_idx"]].append(row["distance"])

        frame_indices = sorted(grouped_frames)
        x_values = [frame / fps for frame in frame_indices] if use_time_axis else frame_indices
        all_x_values = [row["frame_idx"] / fps for row in rows] if use_time_axis else [
            row["frame_idx"] for row in rows
        ]
        all_y_values = [row["distance"] for row in rows]
        median_values = [median(grouped_frames[frame]) for frame in frame_indices]

        plt.scatter(all_x_values, all_y_values, s=8, alpha=0.25, label="Pair observations")
        plt.plot(x_values, median_values, color="black", linewidth=2, label="Per-frame median")

    plt.xlabel("Time (seconds)" if use_time_axis else "Frame number")
    plt.ylabel("Estimated pair distance (relative units)")
    plt.title(title or "Estimated pairwise distance over time")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV, help="Pairwise-distance CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_IMAGE, help="Output PNG path")
    parser.add_argument("--class-name", default="person", help="Keep only pairs of this class")
    parser.add_argument("--all-classes", action="store_true", help="Plot every object-class pair")
    parser.add_argument("--fps", type=float, help="Video FPS; changes the X-axis from frames to seconds")
    parser.add_argument("--show-pairs", action="store_true", help="Draw separate lines for frame-local pair labels")
    args = parser.parse_args()

    object_class = None if args.all_classes else args.class_name
    rows = load_rows(args.input, object_class=object_class)
    plot_rows(rows, args.output, fps=args.fps, show_pairs=args.show_pairs)
    print(f"Plotted {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
