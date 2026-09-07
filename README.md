[detections (2).csv](https://github.com/user-attachments/files/31917121/detections.2.csv)# Human Detector 🎯

A real-time object detection pipeline built with **YOLOv8** and **OpenCV** that processes video files, annotates detected objects, logs results to CSV, and generates a summary detection report. Runs both **locally** (with a live preview window) and in **Google Colab** (with periodic frame previews) using the same script.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen)
![OpenCV](https://img.shields.io/badge/OpenCV-Video%20Processing-red)
![Colab](https://img.shields.io/badge/Google%20Colab-Supported-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📋 Overview

`human_detector` runs a YOLOv8 model over a video file frame-by-frame, drawing bounding boxes and class labels on each detected object. While processing, it:

- Overlays a **live FPS counter** and **per-frame object count** on the video
- Writes an **annotated output video** to disk
- Logs every detection (frame number, class, confidence) to a **CSV file**
- Displays a live preview (a GUI window locally, or throttled inline frames in Colab)
- Prints a **final summary report** with total frames, total objects, and a per-class breakdown

The script auto-detects its environment at runtime, so the exact same `detect.py` works whether you run it on your own machine or in a Colab notebook.

---

## ⚙️ How It Works

1. **Environment detection** — Attempts to import Colab-only modules (`google.colab.patches.cv2_imshow`, `google.colab.files`). If the import succeeds, the script runs in **Colab mode**; otherwise it falls back to **local mode**.
2. **Model loading** — Loads a YOLOv8 model (`yolov8n.pt` by default) via the `ultralytics` library.
3. **Video setup** — Opens the input video with OpenCV (`cv2.VideoCapture`) and configures an `mp4v`-encoded `VideoWriter` for the annotated output, matching the source's resolution and FPS (with a fallback FPS of 30 if the source reports 0, which can happen with some uploaded files in Colab).
4. **CSV logging** — Initializes a `detections.csv` file with the header `Frame, Class, Confidence`.
5. **Frame-by-frame inference loop**:
   - Reads a frame from the video.
   - Runs YOLOv8 inference (`model.predict`) with a configurable confidence threshold.
   - For each detected box: extracts the class ID, confidence score, and class name, updates running statistics, and writes a row to the CSV.
   - Draws bounding boxes/labels on the frame using YOLOv8's built-in `.plot()` method.
   - Overlays the live FPS (computed from elapsed time and frames processed) and the current frame's object count.
   - Writes the annotated frame to the output video.
   - **Local mode:** displays the frame in a live `cv2.imshow()` window; press `q` to stop early.
   - **Colab mode:** displays a frame preview every `SHOW_EVERY_N_FRAMES` frames using `cv2_imshow()`, since Colab has no GUI backend and no keyboard interrupt support mid-loop.
6. **Cleanup** — Releases the video capture and writer, closes the CSV file, and (in local mode) destroys the display window.
7. **Final report** — Prints total frames processed, total objects detected, and a sorted count of detections per class.
8. **Results download (Colab only)** — Automatically triggers a browser download of `output.mp4` and `detections.csv` at the end of the run.

---

## 📦 Requirements

- Python 3.8+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV (`opencv-python`)

Install dependencies locally:

```bash
pip install ultralytics opencv-python
```

In Colab, dependencies are installed with:

```python
!pip install ultralytics -q
```

(`opencv-python` and `google.colab` come pre-installed in the Colab runtime.)

---

## 📁 Project Structure

```
human_detector/
├── video/
│   └── human.mp4          # Input video (place your video here for local runs)
├── detect.py                # Main detection script (local + Colab compatible)
├── requirements.txt          # Python dependencies
├── output.mp4                 # Generated annotated output video
├── detections.csv               # Generated detection log
└── README.md
```

---

## 🚀 Usage

### Option 1: Run Locally

1. Place your input video inside the `video/` folder (or update `VIDEO_PATH` in the script).
2. Ensure the YOLOv8 weights file (`yolov8n.pt`) is available, or leave the default — it auto-downloads on first run.
3. Run the script:

```bash
python detect.py
```

4. A live preview window opens showing detections in real time. Press **`q`** to stop processing early.
5. On completion, check the project root for `output.mp4`, `detections.csv`, and a console summary report.

### Option 2: Run in Google Colab

1. Open a new Colab notebook and set the runtime to **GPU** for best performance (`Runtime → Change runtime type → GPU`).
2. Upload `detect.py` or paste its contents into a cell.
3. Install dependencies:

```python
!pip install ultralytics -q
```

4. Upload your video when prompted (via `files.upload()`), or mount Google Drive and point `VIDEO_PATH` to your file.
5. Run the script. Frame previews render inline every `SHOW_EVERY_N_FRAMES` frames instead of a live window.
6. At the end, `output.mp4` and `detections.csv` are automatically downloaded to your browser.

---

## 🔧 Configuration

All key parameters are defined at the top of the script:

| Variable | Description | Default |
|---|---|---|
| `MODEL_PATH` | Path to the YOLOv8 model weights | `yolov8n.pt` |
| `VIDEO_PATH` | Path to the input video | `video/human.mp4` |
| `OUTPUT_VIDEO` | Path for the annotated output video | `output.mp4` |
| `CONFIDENCE` | Minimum confidence threshold for detections | `0.5` |
| `SHOW_EVERY_N_FRAMES` | How often to render a preview frame in Colab mode | `30` |
| `IN_COLAB` | Auto-detected at runtime; do not set manually | — |

---

## 📊 Output Example

**Console report:**
```
========== REPORT ==========
Frames Processed : 452
Objects Detected : 1187
person: 1187
============================
```

**CSV log (`detections.csv`):**
| Frame | Class  | Confidence |
|-------|--------|------------|
| 1     | person | 0.87       |
| 1     | person | 0.79       |
| 2     | person | 0.91       |

[[Uplo
https://github.com/user-attachments/assets/b5b2e096-b7da-4f35-9763-8c197917cefb
ading detections (2).csv…]()](https://github.com/TARIQ8099/human_detector/tree/main/Output%20Example)


---

## 💡 Local vs. Colab: Key Differences

| Behavior | Local Mode | Colab Mode |
|---|---|---|
| Live preview | `cv2.imshow()` window, every frame | `cv2_imshow()`, every `SHOW_EVERY_N_FRAMES` frames |
| Early exit | Press `q` | Interrupt the runtime manually |
| Video input | Read from local `video/` path | Upload via `files.upload()` or Google Drive |
| Result retrieval | Files saved to local disk | Auto-downloaded via `files.download()` |
| Recommended hardware | CPU is usually fine | GPU runtime strongly recommended for longer videos |

---

## 🛠️ Potential Improvements

- Add support for real-time webcam input (`cv2.VideoCapture(0)`)
- Multi-class filtering (e.g., detect only `person`)
- Export summary report as JSON/PDF alongside CSV
- Add unique object tracking (e.g., with `ByteTrack`/`DeepSORT`) instead of raw per-frame counts
- Command-line arguments (`argparse`) instead of hardcoded configuration

---

## 📄 License

This project is licensed under the MIT License. Feel free to use and modify it for your own purposes.

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection model
- [OpenCV](https://opencv.org/) for video processing and visualization
- [Google Colab](https://colab.research.google.com/) for free GPU-backed notebook execution
