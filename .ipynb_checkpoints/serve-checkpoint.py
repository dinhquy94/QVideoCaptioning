 
# 🎥 Q-ClassCap: Video Captioning for Classroom Behavior Understanding

Q-ClassCap is an end-to-end deep learning system for generating natural language descriptions from classroom video clips. It combines motion, object, and context information with multi-head fusion and a transformer-based captioning model.

---

## 📁 Project Structure

```

├── models/                # Model architecture (ObjectHead, ActionHead, GlobalHead, CaptionHead)
├── dataset/               # Dataset classes (VideoDataset, ValVideoDataset)
├── features/              # Feature extractors (I3D, DETR, CLIP, context)
├── utils/                 # Utilities (collate functions, metrics, tokenizer utils)
├── checkpoints/           # Pretrained or intermediate checkpoints
├── uploads/               # Uploaded video files (for API)
├── train.py               # Training pipeline
├── validate.py            # Validation and evaluation
├── serving.py             # Flask REST API for real-time captioning
└── README.md              # This documentation

````

---

## 🧠 Model Overview

- **Feature Extractors**:
  - Temporal: `VideoMAE` / `VideoMAE` pretrained
  - Object: `DETR` + `CLIP` semantic embedding
  - Context: `CLIP` on keyframes

- **Heads**:
  - `ObjectHead`: Detect salient objects
  - `ActionHead`: Fuse motion-object interactions
  - `GlobalHead`: Global semantic encoder
  - `CaptionHead`: QFormer + Deepseek VL2 decoder for caption generation

- **Training Objective**:
  - Multi-loss: Hungarian loss for object/action, cosine loss for global features, cross-entropy for caption

---

## 🚀 Setup

```bash
git clone https://github.com/your-username/q-classcap.git
cd q-classcap
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

---

## 🏋️‍♂️ Training

```bash
python train.py \
  --dataset_path ./data/vatex/train \
  --feature_dir ./video_features \
  --checkpoint ./checkpoints/init.pt \
  --output_dir ./checkpoints \
  --batch_size 16 \
  --num_epochs 20 \
  --lr 1e-5
```

---

## 🧪 Evaluation

```bash
python validate.py \
  --val_json ./data/val.json \
  --feature_dir ./video_features \
  --batch_size 8 \
  --epoch 7 \
  --checkpoint ./checkpoints/model_checkpoint-epoch-7.pt
```

---

## 🌐 Serving API (Flask)

```bash
python serving.py \
  --model_checkpoint ./checkpoints/model_checkpoint-epoch-7.pt \
  --upload_folder uploads \
  --device cuda \
  --port 5000
```

### POST `/upload`

* **Request**: `multipart/form-data` with a key `video` containing a `.mp4`, `.avi`, etc.
* **Response**: JSON with generated caption

```json
{
  "caption": "A student is sitting and looking at the screen while the teacher walks nearby."
}
```

---

## 📊 Sample Result

| VideoID   | Ground-truth                                | Prediction                                |
| --------- | ------------------------------------------- | ----------------------------------------- |
| `clip001` | "A student uses a phone in class."          | "A student is holding a phone."           |
| `clip002` | "Two students are discussing and pointing." | "Two people are talking near a computer." |

---

## 🛠️ Dependencies

* PyTorch
* Huggingface Transformers
* OpenCLIP
* Flask, Flask-CORS
* scikit-learn, pandas, tqdm

Install all:

```bash
pip install -r requirements.txt
```

---

## 📎 Citation

If you use this code for academic work, please cite:

```bibtex
@misc{qclasscap2025,
  title={Q-ClassCap: An End-to-End Multi-Head Captioning Model for Classroom Behavior Videos},
  author={Nguyen Dinh Quy},
  year={2025},
  url={https://github.com/dinhquy94/q-classcap}
}
```

---

## 📬 Contact

For any questions or collaboration inquiries, please contact: [quynd@huce.edu.vn](mailto:quynd@huce.edu.vn)
 