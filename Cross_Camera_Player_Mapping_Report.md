# 📄 Cross-Camera Player Mapping – Project Report

## 1. Objective

The aim of this project is to detect and match soccer players across two different camera perspectives — `broadcast` and `tacticam` — using a YOLOv8 object detector and ResNet-based appearance embeddings.

---

## 2. Dataset

- **Videos Used:**
  - `broadcast.mp4`
  - `tacticam.mp4`

These videos were provided as part of the challenge and are used to extract frames and detect players and the ball.

---

## 3. Model

- **Object Detector:** YOLOv8  
  A fine-tuned YOLOv8 model (`best.pt`) was provided for detecting `player`, `ball`, `goalkeeper`, and `referee`.

- **Embedding Model:** ResNet-18  
  Pretrained on ImageNet and used to extract 512-dimensional feature vectors for each detected player.

---

## 4. Pipeline

### Step 1: Frame Extraction
- Extract frames from both videos using OpenCV.
- Save them under:
  - `outputs/broadcast_frames/`
  - `outputs/tacticam_frames/`

### Step 2: Player & Ball Detection
- Run the YOLOv8 model on each frame.
- Only bounding boxes for class `"player"` are considered for matching.

### Step 3: Embedding Extraction
- Each detected player is cropped.
- Embeddings are extracted using ResNet-18 and saved as `.npy` files.

### Step 4: Player Matching
- Cosine similarity is computed between player embeddings from broadcast and tacticam frames.
- Hungarian algorithm is used to optimally match players.

### Step 5: Visualization
- Matched bounding boxes are drawn across both camera views.

---

## 5. Folder Structure

```
cross_camera_player_mapping/
├── data/
│   ├── broadcast.mp4
│   └── tacticam.mp4
├── model/
│   └── best.pt
├── outputs/
│   ├── broadcast_frames/
│   ├── tacticam_frames/
│   ├── broadcast_embeddings/
│   ├── tacticam_embeddings/
│   └── match_results/
├── scripts/
│   ├── extract_detections.py
│   ├── extract_detections_tacticam.py
│   ├── extract_embeddings.py
│   ├── extract_embeddings_tacticam.py
│   ├── match_players.py
│   └── visualize_matches.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 6. Key Learnings

- Hands-on experience with the **Ultralytics YOLOv8** framework.
- Learned how to work with **deep embeddings** and **similarity-based matching**.
- Applied **Hungarian algorithm** for optimal bipartite graph matching.
- Developed a full ML pipeline from **video → detection → embedding → matching → visualization**.

