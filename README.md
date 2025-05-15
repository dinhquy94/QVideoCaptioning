 
# 🎬 QVidCap: End-to-End Video Captioning with Multi-Modal Feature Fusion

**QVidCap** is an end-to-end deep learning framework for video captioning. It generates natural language descriptions for short video clips by integrating motion, object, and context features using a multi-head attention-based architecture and a powerful language decoder.

---

## 📂 Project Structure

```

├── models/                # Model architecture (ObjectHead, ActionHead, GlobalHead, CaptionHead)
├── dataset/               # Dataset loaders and preprocessing (VATEX format)
├── features/              # Feature extraction (VideoMAE, DETR, CLIP)
├── utils/                 # Utilities (collate functions, evaluation metrics, tokenizer utils)
├── checkpoints/           # Saved model checkpoints
├── uploads/               # Upload folder for API video inputs
├── train.py               # Training pipeline
├── validate.py            # Evaluation script
├── serving.py             # Flask API for real-time caption generation
└── README.md              # This file

````

---

## 🧠 Model Architecture

### 🔍 Feature Extractors

| Type      | Backbone       | Description                                      |
|-----------|----------------|--------------------------------------------------|
| Temporal  | [VideoMAE](https://github.com/MCG-NJU/VideoMAE) | Captures motion features across time |
| Object    | DETR + CLIP    | Object detection with semantic embeddings        |
| Context   | CLIP (ViT-B/32)| Visual context extracted from keyframes          |


### 🧩 Multi-Head Modules

- **ObjectHead**: Learns attention over object-centric regions.
- **ActionHead**: Fuses motion-object interaction features.
- **GlobalHead**: Encodes holistic video representation.
- **CaptionHead**: Uses QFormer + [DeepseekVL2](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny) decoder to generate captions.

For generating natural language descriptions, we use the [**Deepseek-VL2**](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny) model as the language decoder. It is a vision-language pretrained transformer that supports both image-text and video-text tasks.

Instead of using traditional autoregressive decoders like GPT-2, Deepseek-VL2 enables:

* **Stronger alignment** with visual features (especially in multi-modal fusion).
* **Pretrained video-text grounding**, improving zero-shot or few-shot generalization.
* **QFormer-style architecture** to efficiently retrieve cross-modal semantic context.

In our architecture, a lightweight **QFormer** queries the fused object, action, and context features. These query outputs are projected and used as **prefix embeddings** for the Deepseek-VL2 decoder, which then generates captions token by token.

> Note: During training, Deepseek-VL2 is frozen to reduce memory usage and prevent overfitting on small video datasets.
 
---

## 📦 Dataset: [VATEX](https://research.google.com/ava/download.html)

This project uses the **VATEX dataset**, which contains multilingual video descriptions for over 41,000 YouTube videos.

### 📥 Download from Kaggle

You can download preprocessed versions of the VATEX dataset from Kaggle:

* [`vatex011011`](https://www.kaggle.com/datasets/khaledatef1/vatex011011)
* [`vatex0110`](https://www.kaggle.com/datasets/khaledatef1/vatex0110)
* [`vatex01101`](https://www.kaggle.com/datasets/khaledatef1/vatex01101)

After downloading, unzip the dataset:

```bash
unzip vatex0110.zip -d dataset_vatex/
```

And structure it like:

```
dataset_vatex/
├── vatex0110/
│   ├── versions/1/vatex/json/train
│   ├── versions/1/vatex/json/public_test
│   └── ...
```
 

### Folder Structure (Expected)

```
dataset_vatex/
├── train/                   # Training JSON
├── public_test/             # Public test JSON
├── versions/1/vatex/json/   # JSON captions (VATEX-style)
```

---

## 🚀 Setup

```bash
git clone https://github.com/dinhquy94/QVideoCaptioning.git
cd vidcap
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Setup DeepSeek-VL2

```bash
git clone https://github.com/deepseek-ai/DeepSeek-VL2.git
cd DeepSeek-VL2
pip install -e .
```
---

## 🏋️‍♂️ Training

```bash
python train.py \
  --dataset_path ./dataset_vatex/versions/1/vatex/json/train \
  --feature_dir ./video_features_vatex_MAE_v2 \
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
  --test_data_path ./dataset_vatex/versions/1/vatex/json/public_test \
  --val_feature_dir ./video_features_vatex_MAE_v2 \
  --val_batch_size 8 \
  --epoch 7 \
  --val_checkpoint ./checkpoints/model_checkpoint-epoch-7.pt
```

---

## 🌐 Real-Time Inference via API

Start the Flask-based REST API:

```bash
python serving.py \
  --model_checkpoint ./checkpoints/model_checkpoint-epoch-7.pt \
  --upload_folder uploads \
  --device cuda \
  --port 5000
```

### ▶️ POST `/upload`

* **Input**: A video file (MP4/AVI/MKV/etc.) via `multipart/form-data`
* **Response**:

```json
{
  "caption": "A woman is walking into the room while others are sitting at computers."
}
```

---

## 📊 Example Output

| Video ID   | Predicted Caption                           |
| ---------- | ------------------------------------------- |
| `video001` | A group of people are sitting at a table.   |
| `video002` | A man is riding a motorcycle in the street. |

---

## 📦 Dependencies

Install all required dependencies:

```bash
pip install -r requirements.txt
```

Includes:

* PyTorch
* HuggingFace Transformers
* OpenCLIP
* Flask / Flask-CORS
* Kaggle API
* scikit-learn, pandas, tqdm

---

## 📌 Citation

If you use this repository for your research or development, please cite:

```bibtex
@misc{vidcap2025,
  title={VidCap: End-to-End Video Captioning with Multi-Modal Feature Fusion},
  author={Nguyen Dinh Quy},
  year={2025},
  url={https://github.com/dinhquy94/QVideoCaptioning}
}
```

---

## 📬 Contact

For questions, issues, or collaborations, please reach out to: [quynd@huce.edu.vn](mailto:quynd@huce.edu.vn)
 
