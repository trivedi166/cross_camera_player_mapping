# Cross-Camera Player Mapping 🏃📹

This project detects and matches football players across two camera perspectives — a **broadcast view** and a **tacticam view**. It leverages a fine-tuned YOLOv8 model for player detection and ResNet-based embeddings for appearance matching using cosine similarity.

---

## 📁 Project Structure

```
cross_camera_player_mapping/
├── data/                      # Contains input videos
│   ├── broadcast.mp4
│   └── tacticam.mp4
├── model/                     # YOLOv8 model
│   └── best.pt
├── outputs/                   # All generated outputs
│   ├── broadcast_frames/
│   ├── tacticam_frames/
│   ├── broadcast_embeddings/
│   ├── tacticam_embeddings/
│   └── match_results/
├── scripts/                   # All scripts used
│   ├── extract_detections.py
│   ├── extract_detections_tacticam.py
│   ├── extract_embeddings.py
│   ├── extract_embeddings_tacticam.py
│   ├── match_players.py
│   └── visualize_matches.py
├── requirements.txt
└── README.md
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone <git@github.com:trivedi166/cross_camera_player_mapping.git>
cd cross_camera_player_mapping
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure your data/` folder contains both videos (`broadcast.mp4` and `tacticam.mp4`).

### 📦 Download Model

Download the YOLOv8 model weights manually:

- File: `best.pt`
- Place it in the `model/` directory

> **Note**: The model file is not included in this repo due to GitHub’s 100MB file size limit.


---

## 🚀 Running the Pipeline

### Step 1: Extract Detections (YOLOv8)
```bash
python3 scripts/extract_detections.py
python3 scripts/extract_detections_tacticam.py
```

### Step 2: Extract Embeddings (ResNet18)
```bash
python3 scripts/extract_embeddings.py
python3 scripts/extract_embeddings_tacticam.py
```

### Step 3: Match Players Using Cosine Similarity
```bash
python3 scripts/match_players.py
```

### Step 4: Visualize Matches
Update the `FRAME_ID` variable in `visualize_matches.py`, then run:
```bash
python3 scripts/visualize_matches.py
```

---

## 📦 Dependencies

Installed via `requirements.txt`:
- `ultralytics`
- `torch`
- `torchvision`
- `opencv-python`
- `numpy`
- `matplotlib`
- `Pillow`
- `scikit-learn`

---

## 🧠 Approach

- **YOLOv8** is used to detect players and balls in each frame.
- **ResNet18** (pretrained) extracts visual features (embeddings) from each detected player.
- **Cosine distance** and the **Hungarian algorithm** align players across camera views.
- Matching is done frame-by-frame.

---

## ✅ Deliverables

- ✔️ Working and modular codebase
- ✔️ Frame-wise player embeddings
- ✔️ Matching logic via cosine similarity
- ✔️ Visualization of matched players across views

---

## ✍️ Author

**Abhinav Trivedi**  
[LinkedIn](www.linkedin.com/in/abhinav-trivedi-04526b123)  
[GitHub](https://github.com/trivedi166)

---

## 📌 Notes

- Ball detection is included but not used for embedding or matching.
