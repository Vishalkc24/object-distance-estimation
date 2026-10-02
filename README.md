# Object Distance Estimation

This project detects objects in a video with YOLOv8 and estimates monocular depth with Depth Anything V2. It writes an annotated video, per-detection CSV data, and pairwise distance estimates.

## Setup

Use Python 3. Install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The scripts download the Depth Anything V2 model from Hugging Face on first use. The YOLOv8 weights are expected at `yolov8n.pt`.

## Run the pipeline

Place a video named `input.mp4` in the project directory, then run:

```bash
python3 detect_and_measure_depth.py
python3 compute_pairwise_distances.py
python3 plot_pairwise_distances.py
```

Outputs include:

- `output_with_depth.mp4`: annotated video
- `detections_with_depth.csv`: per-object detections and sampled depth
- `pairwise_distances.csv`: within-frame pairwise estimates
- `pairwise_distances_plot.png`: labeled distance chart

## Notes

The standard Depth Anything V2 Small model produces relative depth, not metric depth. Therefore, pairwise distances are relative estimates unless a metric-depth model or external scale calibration is used. Detection indexes are frame-local because the current pipeline does not perform object tracking.
