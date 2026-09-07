"""
human_detector - Colab-compatible version
Real-time object detection using YOLOv8 + OpenCV, adapted to run in
Google Colab notebooks (no local GUI window support).
"""

# ==========================
# IMPORTS
# ==========================
from ultralytics import YOLO
import cv2
import time
import os
import csv
from collections import defaultdict

# The two imports below are Colab-only. If you plan to also run this
# script locally, wrap them in a try/except as shown.
try:
    from google.colab.patches import cv2_imshow
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ==========================
# CONFIGURATION
# ==========================
MODEL_PATH = "yolov8n.pt"          # auto-downloads on first use
VIDEO_PATH = "video/human.mp4"     # update this or upload via files.upload()
OUTPUT_VIDEO = "output.mp4"
CONFIDENCE = 0.5
SHOW_EVERY_N_FRAMES = 30           # throttle preview frequency in Colab

# ==========================
# LOAD MODEL
# ==========================
model = YOLO(MODEL_PATH)

# ==========================
# STATISTICS
# ==========================
total_frames = 0
total_objects = 0
class_counter = defaultdict(int)

# ==========================
# VIDEO SETUP
# ==========================
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(5)) or 30   # fallback if source FPS reads as 0

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# ==========================
# CSV LOGGER
# ==========================
csv_file = open("detections.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Class", "Confidence"])

# ==========================
# PROCESS VIDEO
# ==========================
start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    total_frames += 1

    results = model.predict(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )

    boxes = results[0].boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls]
        class_counter[class_name] += 1
        total_objects += 1
        csv_writer.writerow([
            total_frames,
            class_name,
            round(conf, 2)
        ])

    annotated = results[0].plot()

    current_time = time.time()
    fps_live = total_frames / (current_time - start_time)

    cv2.putText(
        annotated,
        f"FPS: {fps_live:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.putText(
        annotated,
        f"Objects: {len(boxes)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    writer.write(annotated)

    # Colab has no GUI window - preview periodically instead
    if IN_COLAB and total_frames % SHOW_EVERY_N_FRAMES == 0:
        cv2_imshow(annotated)
    elif not IN_COLAB:
        cv2.imshow("YOLOv8 Advanced Detector", annotated)
        if cv2.waitKey(1) == ord("q"):
            break

# ==========================
# CLEANUP
# ==========================
cap.release()
writer.release()
csv_file.close()
if not IN_COLAB:
    cv2.destroyAllWindows()

# ==========================
# FINAL REPORT
# ==========================
print("\n========== REPORT ==========")
print(f"Frames Processed : {total_frames}")
print(f"Objects Detected : {total_objects}")
for cls, count in sorted(class_counter.items()):
    print(f"{cls}: {count}")
print("============================")

# ==========================
# DOWNLOAD RESULTS (Colab only)
# ==========================
if IN_COLAB:
    files.download(OUTPUT_VIDEO)
    files.download("detections.csv")
