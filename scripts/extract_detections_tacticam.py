import cv2
from ultralytics import YOLO
import os

# === CONFIG ===
MODEL_PATH = "/home/abhi169/cross_camera_player_mapping/model/best.pt"
VIDEO_PATH = "/home/abhi169/cross_camera_player_mapping/data/tacticam.mp4"

OUTPUT_DIR = "../outputs/tacticam_frames"
CONFIDENCE_THRESHOLD = 0.5

# === CREATE OUTPUT FOLDER ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD MODEL ===
print("[INFO] Loading model...")
model = YOLO(MODEL_PATH)
print("[INFO] Model loaded successfully.")

print("[DEBUG] Class labels:", model.names)

# === READ VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("[ERROR] Could not open video:", VIDEO_PATH)
    exit()

frame_idx = 0
print("[INFO] Starting video frame-by-frame processing...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video reached or cannot read frame.")
        break

    print(f"[DEBUG] Processing frame {frame_idx}")
    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{model.names[cls_id]} {conf:.2f}"
        color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(f"{OUTPUT_DIR}/frame_{frame_idx:05}.jpg", frame)
    frame_idx += 1

cap.release()
print(f"[INFO] Processed {frame_idx} frames. Output saved to {OUTPUT_DIR}")
