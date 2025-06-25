# scripts/match_players.py

import os
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

BROADCAST_EMB_DIR = "/home/abhi169/cross_camera_player_mapping/outputs/broadcast_embeddings"
TACTICAM_EMB_DIR = "/home/abhi169/cross_camera_player_mapping/outputs/tacticam_embeddings"
MATCH_OUTPUT_DIR = "/home/abhi169/cross_camera_player_mapping/outputs/match_results"


os.makedirs(MATCH_OUTPUT_DIR, exist_ok=True)

frame_files = sorted(f for f in os.listdir(BROADCAST_EMB_DIR) if f.endswith(".npy"))

for frame_file in frame_files:
    broadcast_path = os.path.join(BROADCAST_EMB_DIR, frame_file)
    tacticam_path = os.path.join(TACTICAM_EMB_DIR, frame_file)

    if not os.path.exists(tacticam_path):
        print(f"[WARN] Missing tacticam frame {frame_file}, skipping.")
        continue

    emb_broadcast = np.load(broadcast_path)
    emb_tacticam = np.load(tacticam_path)

    if emb_broadcast.shape[0] == 0 or emb_tacticam.shape[0] == 0:
        print(f"[INFO] No players in frame {frame_file}, skipping.")
        continue

    # Compute cosine distances (smaller = more similar)
    dist_matrix = cdist(emb_broadcast, emb_tacticam, metric='cosine')

    # Match using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    matches = []
    for i, j in zip(row_ind, col_ind):
        similarity = 1 - dist_matrix[i][j]
        matches.append((i, j, round(similarity, 3)))

    # Save results
    out_path = os.path.join(MATCH_OUTPUT_DIR, frame_file.replace(".npy", ".txt"))
    with open(out_path, "w") as f:
        for i, j, sim in matches:
            f.write(f"broadcast_player_{i} <-> tacticam_player_{j} | similarity: {sim}\n")

    print(f"[INFO] Matched {len(matches)} players in {frame_file}")
