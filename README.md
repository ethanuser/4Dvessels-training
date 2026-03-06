# 4D Vessels — Training Pipeline

This repository contains the **preprocessing pipeline** for the 4D vessel imaging project. Starting from multi-camera OBS recordings, the scripts in this repo produce a folder of split individual-camera videos that are ready to be fed into the SAM2 segmentation code (which runs on the GPU).

---

## Prerequisites

Before using this repo, you should already have:

1. **OBS recording(s)** of the vessel experiment (multi-camera grid video, `.mkv`)
2. **Calibration video** recorded from the same camera setup (checkerboard visible in all cameras)
3. **Python 3.9+** installed on your machine
4. **FFmpeg** installed and available on your system PATH (used for video trimming and splitting)

### Installing FFmpeg

- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin/` folder to your system PATH.
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

Verify installation:
```bash
ffmpeg -version
```

---

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-ORG/4Dvessels-training.git
cd 4Dvessels-training
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

- **Windows:** `venv\Scripts\activate`
- **macOS / Linux:** `source venv/bin/activate`

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: `numpy`, `opencv-python`, `pandas`, `matplotlib`, `scipy`, `tqdm`, `Pillow`, and `pyvista`.

---

## Repository Structure

```
4Dvessels-training/
├── Preprocess/
│   ├── config.json                          # Central configuration file
│   ├── bounding_boxes.json                  # Bounding box data for cropping
│   ├── Camera Calibration/
│   │   ├── create_calibration_images.py     # Step 1: Extract calibration frames
│   │   ├── gen_calib_data_images.py         # Step 2: Run camera calibration
│   │   ├── plot_camera_positions.py         # Utility: Visualize camera positions
│   │   ├── plot_camera_views.py             # Utility: 3D camera view visualization
│   │   └── calibrations/                    # Output: calibration images & parameters
│   ├── Filter Background/
│   │   ├── trim_video.py                    # Trim raw OBS video to experiment window
│   │   └── split_videos.py                  # Split grid video into per-camera videos
│   ├── Timestamping/
│   │   └── gen_timestamps.py                # Generate normalized timestamps
│   └── Create Nerf Datasets/
│       └── create_nerf_data_full.py         # Create NeRF-format dataset
├── NeRF Data Upload/                        # Output directory for NeRF datasets
├── requirements.txt
└── README.md
```

---

## Pipeline Overview

The full workflow produces **split per-camera videos** from a multi-camera grid recording. These split videos are then taken to the SAM2 GPU machine for segmentation.

```
OBS Grid Video → Trim → Timestamps → Split into per-camera videos → (SAM2 on GPU)
                                          ↑
                  Camera Calibration ──────┘ (needed for NeRF dataset creation)
```

---

## Step-by-Step Usage

### Step 1: Configure `config.json`

Open `Preprocess/config.json` and update the paths to match your local machine:

```json
{
    "video_path": "path/to/your/experiment_trimmed_ds_N.mkv",
    "timestamps_path": "path/to/this/repo/Preprocess/Timestamping/timestamps.csv",
    "calibration_video_path": "path/to/your/calibration_video.mkv",
    "calibration_path": "path/to/this/repo/Preprocess/Camera Calibration/calibrations/camera_parameters.json",
    "grid_rows": 3,
    "grid_cols": 3,
    "view_width": 1360,
    "view_height": 1360,
    "num_cams": 9,
    ...
}
```

> **Note:** All file paths in `config.json` use `\\` (double backslash) on Windows. On macOS/Linux, use forward slashes `/`.

### Step 2: Camera Calibration

These steps process the checkerboard calibration video to determine camera positions and orientations.

#### 2a. Extract Calibration Images

Set `calibration_video_path` in `config.json` to point to your calibration video, then run:

```bash
python Preprocess/Camera\ Calibration/create_calibration_images.py
```

This extracts the first frame of the calibration video and splits it into individual camera views. Output is saved to:
```
Preprocess/Camera Calibration/calibrations/
```

Verify that all camera images (e.g., `view_1.png` through `view_8.png`) have been saved.

#### 2b. Run Camera Calibration

```bash
python Preprocess/Camera\ Calibration/gen_calib_data_images.py
```

This detects checkerboard corners in each image and computes camera positions/orientations. On success, a 3D graph will appear showing the camera arrangement — verify it matches your physical setup (e.g., 4 cameras on top, 4 on bottom).

**If corner detection fails** for any camera, try:
- Re-recording the calibration video with better lighting / sharper focus
- Adjusting the contrast/brightness constants at the top of `create_calibration_images.py`

**If camera orientations are wrong** (corner #1 not at the blue dot position), adjust the flip arrays in `gen_calib_data_images.py`:
```python
flip_lr = [0, 0, 0, 0, 0, 0, 0, 0]  # 1 = flip left/right
flip_up = [0, 0, 0, 0, 0, 0, 0, 0]  # 1 = flip up/down
```
Re-run until the XYZ graph matches the real-world layout.

### Step 3: Trim the Video

Open `Preprocess/Filter Background/trim_video.py` and update the variables at the top:

```python
VIDEO_PATH = r"path/to/your/experiment.mkv"
OUTPUT_PATH = r"path/to/your/experiment_trimmed.mkv"
START_TIME = 2    # Start time in seconds
DURATION = 14     # Duration of clip in seconds (not end time)
```

> **Tip:** Use VLC to scrub through the video and identify the start/end of the deformation event. Add a small buffer before and after.

Then run:

```bash
python Preprocess/Filter\ Background/trim_video.py
```

### Step 4: Generate Timestamps

Update `video_path` in `config.json` to point to your trimmed (and optionally downsampled) video, then run:

```bash
python Preprocess/Timestamping/gen_timestamps.py
```

This generates a CSV of normalized timestamps (0.0 to 1.0) for each frame, saved to `Preprocess/Timestamping/timestamps.csv`.

### Step 5: Split Video into Per-Camera Videos

Open `Preprocess/Filter Background/split_videos.py` and update:

```python
video_path = r"path/to/your/experiment_trimmed.mkv"
```

Also update `CAMERA_SAVE_FLAGS` to select which cameras to split out (1 = save, 0 = skip):

```python
CAMERA_SAVE_FLAGS = [1, 1, 1, 1, 1, 1, 1, 1, 0]  # Save cameras 1-8, skip 9
```

Then run:

```bash
python Preprocess/Filter\ Background/split_videos.py
```

Output: Individual camera videos saved to a `split_cams/` folder.

---

## Deliverable

After completing the pipeline, you will have a folder of **split per-camera videos**. These are the input for the **SAM2 segmentation** code, which runs on the GPU.

---

## Creating the NeRF Dataset (After SAM2)

After SAM2 masking and merging is complete (done on the GPU machine), return to this repo and run:

```bash
python Preprocess/Create\ Nerf\ Datasets/create_nerf_data_full.py
```

Make sure `config.json` has the correct `video_path` (pointing to the masked/stitched video), `calibration_path`, and `timestamps_path` set. Output will be saved to the `NeRF Data/` directory containing:

- `train/` — extracted and calibrated frames for all cameras
- `transforms_train.json` — camera transform matrices and timestamps

---

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `plot_camera_positions.py` | Visualize camera positions from calibration data (matplotlib 3D plot) |
| `plot_camera_views.py` | Interactive 3D visualization of camera positions with image textures (PyVista) |

> **Note:** These utility scripts have hardcoded file paths that need to be updated to your local paths before running.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ffmpeg` / `ffprobe` not found | Install FFmpeg and add it to your system PATH |
| Checkerboard corners not detected | Improve lighting, adjust focus, or tweak contrast constants in `create_calibration_images.py` |
| Camera positions don't match physical setup | Adjust `flip_lr` / `flip_up` arrays in `gen_calib_data_images.py` |
| Timestamp count mismatch | Ensure `config.json` video_path points to the same video used for timestamp generation |
| `config.json` path errors on macOS/Linux | Replace `\\` with `/` in all file paths |
