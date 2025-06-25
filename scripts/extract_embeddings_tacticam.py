# scripts/extract_embeddings.py

import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = "/home/abhi169/cross_camera_player_mapping/model/best.pt"

FRAMES_DIR = "../outputs/tacticam_frames"
EMBEDDING_OUTPUT_DIR = "../outputs/tacticam_embeddings"
CONFIDENCE_THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === SETUP ===
os.makedirs(EMBEDDING_OUTPUT_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)
print("[INFO] YOLO model loaded.")

# Load ResNet18 feature extractor (remove final classification layer)
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(DEVICE)
resnet.eval()
print("[INFO] Feature extractor (ResNet18) loaded.")

# Image transform for ResNet
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === PROCESS FRAMES ===
frame_files = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg"))

for frame_file in frame_files:
    frame_path = os.path.join(FRAMES_DIR, frame_file)
    frame = cv2.imread(frame_path)
    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    embeddings = []

    for box in boxes:
        cls_id = int(box.cls[0])
        if model.names[cls_id] != "player":
            continue  # Only extract for players

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Preprocess and convert to tensor
        try:
            input_tensor = transform(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                embedding = resnet(input_tensor).cpu().numpy()[0]  # shape: (512,)
            embeddings.append(embedding)
        except Exception as e:
            print(f"[WARNING] Skipping a crop due to error: {e}")

    # Save embeddings
    out_path = os.path.join(EMBEDDING_OUTPUT_DIR, frame_file.replace(".jpg", ".npy"))
    np.save(out_path, np.array(embeddings))
    print(f"[INFO] Saved {len(embeddings)} embeddings -> {out_path}")
