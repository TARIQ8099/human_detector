# Human Detector 🎯

A real-time object detection pipeline built with **YOLOv8** and **OpenCV** that processes video files, annotates detected objects, logs results to CSV, and generates a summary detection report.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen)
![OpenCV](https://img.shields.io/badge/OpenCV-Video%20Processing-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📋 Overview

`human_detector` runs a YOLOv8 model over a video file frame-by-frame, drawing bounding boxes and class labels on each detected object. While processing, it:

- Overlays a **live FPS counter** and **per-frame object count** on the video
- Writes an **annotated output video** to disk
- Logs every detection (frame number, class, confidence) to a **CSV file**
- Displays the processing feed in a live preview window
- Prints a **final summary report** with total frames, total objects, and a per-class breakdown

---

## ⚙️ How It Works

1. **Model loading** — Loads a YOLOv8 model (`yolov8n.pt` by default) via the `ultralytics` library.
2. **Video setup** — Opens the input video with OpenCV (`cv2.VideoCapture`) and configures an `mp4v`-encoded `VideoWriter` for the annotated output, matching the source's resolution and FPS.
3. **CSV logging** — Initializes a `detections.csv` file with the header `Frame, Class, Confidence`.
4. **Frame-by-frame inference loop**:
   - Reads a frame from the video.
   - Runs YOLOv8 inference (`model.predict`) with a configurable confidence threshold.
   - For each detected box: extracts the class ID, confidence score, and class name, updates running statistics, and writes a row to the CSV.
   - Draws bounding boxes/labels on the frame using YOLOv8's built-in `.plot()` method.
   - Overlays the live FPS (computed from elapsed time and frames processed) and the current frame's object count.
   - Writes the annotated frame to the output video and displays it in a window.
   - Exits early if the user presses `q`.
5. **Cleanup** — Releases the video capture and writer, closes the CSV file, and destroys the display window.
6. **Final report** — Prints total frames processed, total objects detected, and a sorted count of detections per class.

---

## 📦 Requirements

- Python 3.8+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV (`opencv-python`)

Install dependencies:

```bash
pip install ultralytics opencv-python
```

---

## 📁 Project Structure

```
human_detector/
├── video/
│   └── human.mp4          # Input video (place your video here)
├── yolov8n.pt              # YOLOv8 model weights
├── main.py                 # Main detection script
├── output.mp4               # Generated annotated output video
├── detections.csv            # Generated detection log
└── README.md
```

---

## 🚀 Usage

1. Place your input video inside the `video/` folder (or update `VIDEO_PATH` in the script).
2. Ensure the YOLOv8 weights file (`yolov8n.pt`) is in the project root, or point `MODEL_PATH` to your preferred model.
3. Run the script:

```bash
python main.py
```

4. A live preview window will open showing detections in real time. Press **`q`** to stop processing early.
5. On completion, check the project root for:
   - `output.mp4` — the annotated video
   - `detections.csv` — a full detection log
   - A console summary report of total frames, objects, and per-class counts

---

## 🔧 Configuration

All key parameters are defined at the top of the script:

| Variable | Description | Default |
|---|---|---|
| `MODEL_PATH` | Path to the YOLOv8 model weights | `yolov8n.pt` |
| `VIDEO_PATH` | Path to the input video | `video/human.mp4` |
| `OUTPUT_VIDEO` | Path for the annotated output video | `output.mp4` |
| `CONFIDENCE` | Minimum confidence threshold for detections | `0.5` |

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
