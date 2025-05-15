# Video Captioning for Classroom Behavior Analysis

This repository provides an end-to-end multi-head video captioning framework for analyzing classroom behaviors using deep learning. The system integrates object detection, action recognition, and global contextual encoding with a transformer-based language model to generate descriptive captions for classroom activity videos.

## 🚀 Features

* **Modular Architecture**: Separate object, action, global, and caption heads.
* **QFormer**: A query-based multi-head attention module for aligning object, action, and context features.
* **Semantic Decoding**: Caption generation using Deepseek-VL2 or GPT-2.
* **Custom Dataset Support**: Easily pluggable for custom video datasets.
* **Hungarian + Cosine Loss**: Structured loss computation for object and action alignment.
* **Evaluation Metrics**: Supports BLEU, CIDEr, METEOR, ROUGE-L.

---

## 📁 Repository Structure

```bash
video-captioning/
├── models/                # Model components: ObjectHead, ActionHead, GlobalHead, QFormer, CaptionHead
├── dataset/               # VideoDataset and ValVideoDataset
├── utils/                 # Utilities: collate_fn, evaluation, caption decoding
├── train.py               # Training script (parametrized)
├── validate.py            # Evaluation script (parametrized)
├── configs/               # (Optional) Configuration YAMLs
├── checkpoints/           # Saved model checkpoints
├── video_features/        # Extracted .pt features for videos
├── video_data/            # JSON annotation files
└── README.md              # Project documentation
```

---

## ⚖️ Requirements

```bash
pip install -r requirements.txt
```

Recommended dependencies:

* `torch`
* `transformers`
* `datasets`
* `tqdm`
* `scikit-learn`
* `evaluate`

---

## 🔧 Training

```bash
python train.py \
    --dataset_path ../dataset_vatex/vatex0110/versions/1/vatex/json/train \
    --feature_dir ../video_features_vatex_MAE_v2 \
    --checkpoint ../model_deepseek_epoc2.pt \
    --output_dir ./checkpoints \
    --batch_size 16 \
    --num_epochs 20 \
    --lr 1e-5
```

---

## ✅ Validation

```bash
python validate.py \
    --val_json ../video_short/caption/val/val.json \
    --feature_dir ../video_features_classroom \
    --checkpoint ./checkpoints/model_checkpoint-epoch-2.pt \
    --batch_size 8 \
    --epoch 2
```

---

## 🏛️ Model Components

* `ObjectHead`: Extracts important object-level features with Transformer Encoder and Multihead Attention.
* `ActionHead`: Attends to object and motion context for action representation.
* `GlobalHead`: Fuses object, action, and motion into a global video representation.
* `QFormer`: Query-driven fusion of semantic features.
* `CaptionHead`: Generates captions using pretrained language models like DeepSeek-VL2 or GPT-2.

---

## 📈 Evaluation

* Predictions vs. References are compared using:

  * BLEU (n-gram match)
  * CIDEr (consensus-based)
  * METEOR (semantic and synonym-aware)
  * ROUGE-L (longest common subsequence)

Metrics are logged per epoch and saved to file for tracking.

---

## 🛠️ Customization

* Replace feature extractor with your own (e.g., I3D, VideoMAEv2)
* Modify `ValVideoDataset` and `VideoDataset` to suit your annotation format
* Switch decoder from DeepSeek to GPT-2 or LLaMA
* Fine-tune QFormer to use cross-modal attention

---

## 🚫 Limitations

* Requires pre-extracted video features
* Performance sensitive to caption quality and label consistency
* Not real-time (batch inference only)

---

## 🌟 Acknowledgments

This project was built on top of:

* [HuggingFace Transformers](https://github.com/huggingface/transformers)
* [VideoMAEv2](https://github.com/OpenGVLab/VideoMAE)
* [DeepSeek-VL](https://github.com/deepseek-ai/DeepSeek-VL)

---

## ✉️ Contact

For feedback or collaboration, please contact: `quynd@huce.edu.vn`
