# scripts/visualize_matches.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
FRAME_ID = "frame_00009"
BROADCAST_IMG = f"/home/abhi169/cross_camera_player_mapping/outputs/broadcast_frames/{FRAME_ID}.jpg"
TACTICAM_IMG = f"/home/abhi169/cross_camera_player_mapping/outputs/tacticam_frames/{FRAME_ID}.jpg"
BROADCAST_BOXES = f"/home/abhi169/cross_camera_player_mapping/outputs/broadcast_embeddings/{FRAME_ID}_boxes.npy"
TACTICAM_BOXES = f"/home/abhi169/cross_camera_player_mapping/outputs/tacticam_embeddings/{FRAME_ID}_boxes.npy"
MATCH_FILE = f"/home/abhi169/cross_camera_player_mapping/outputs/match_results/{FRAME_ID}.txt"


# === LOAD EVERYTHING ===
img_broadcast = cv2.imread(BROADCAST_IMG)
img_tacticam = cv2.imread(TACTICAM_IMG)
img_broadcast = cv2.cvtColor(img_broadcast, cv2.COLOR_BGR2RGB)
img_tacticam = cv2.cvtColor(img_tacticam, cv2.COLOR_BGR2RGB)

boxes_broadcast = np.load(BROADCAST_BOXES)
boxes_tacticam = np.load(TACTICAM_BOXES)

# Read matches
matches = []
with open(MATCH_FILE, "r") as f:
    for line in f:
        parts = line.strip().split("|")
        left, right = parts[0].split("<->")
        b_id = int(left.strip().split("_")[-1])
        t_id = int(right.strip().split("_")[-1])
        sim = float(parts[1].split(":")[-1])
        matches.append((b_id, t_id, sim))

# === VISUALIZE ===
for i, (b_idx, t_idx, sim) in enumerate(matches):
    x1, y1, x2, y2 = boxes_broadcast[b_idx]
    crop_b = img_broadcast[y1:y2, x1:x2]

    x1, y1, x2, y2 = boxes_tacticam[t_idx]
    crop_t = img_tacticam[y1:y2, x1:x2]

    fig, axs = plt.subplots(1, 2, figsize=(6, 4))
    axs[0].imshow(crop_b)
    axs[0].set_title(f"Broadcast {b_idx}")
    axs[1].imshow(crop_t)
    axs[1].set_title(f"Tacticam {t_idx}")

    for ax in axs:
        ax.axis("off")
    fig.suptitle(f"Similarity: {sim:.3f}", fontsize=12)
    plt.tight_layout()
    plt.show()
