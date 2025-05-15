import cv2
import numpy as np
import torch
from torchvision import models, transforms
from torchvision.models.video import r3d_18  # Pretrained 3D-CNN model
from transformers import DetrForObjectDetection
from PIL import Image
import matplotlib.patches as patches
import torchvision.transforms as T
import open_clip
import matplotlib.pyplot as plt

device = 'cuda'


def extract_clip_features(image_list, clip_model):
    """
    Trích xuất đặc trưng CLIP từ danh sách ảnh có kích thước khác nhau.

    Args:
        image_list (List[torch.Tensor]): Danh sách các tensor ảnh với kích thước khác nhau (C, H, W).

    Returns:
        torch.Tensor: Tensor (N, 512) chứa vector đặc trưng của mỗi ảnh.
    """
    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    # Chuẩn hóa từng ảnh trong danh sách
    image_batch = torch.stack([transform(img) for img in image_list])  # (N, 3, 224, 224)

    # Trích xuất đặc trưng
    # print(clip_model)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image_batch)

    # Chuẩn hóa vector đặc trưng
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    return image_features  # (N, 512)

def extract_frames(video_path, N, M):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    key_frame_indices = np.linspace(0, frame_count - 1, N, dtype=int)

    key_frames = np.zeros((N, height, width, 3), dtype=np.uint8)
    surrounding_frames = np.zeros((N, M, height, width, 3), dtype=np.uint8)

    for i, idx in enumerate(key_frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, key_frame = cap.read()
        if ret:
            key_frames[i] = key_frame

        start_idx = max(0, idx - M // 2)
        end_idx = min(frame_count, idx + M // 2)

        temp_frames = []
        for j in range(start_idx, end_idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, j)
            ret, frame = cap.read()
            if ret:
                temp_frames.append(frame)

        while len(temp_frames) < M:
            temp_frames.append(temp_frames[-1] if temp_frames else key_frame)

        surrounding_frames[i] = np.array(temp_frames[:M])

    cap.release()
    return key_frames, surrounding_frames

 

@torch.no_grad()
def extract_context_vector(key_frames, clip_model):
    transform = transforms.ToTensor()
    tensor_frames = [transform(Image.fromarray(f)).to(device) for f in key_frames]
    return extract_clip_features(tensor_frames, clip_model)

@torch.no_grad()
def extract_temporal_features(surrounding_frames, model):
    N = len(surrounding_frames)
    M = len(surrounding_frames[0])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    batch = []
    for clip in surrounding_frames:
        clip_tensor = [transform(Image.fromarray(f)) for f in clip]
        clip_tensor = torch.stack(clip_tensor).to(device)  # (M, 3, H, W)
        batch.append(clip_tensor)
    batch_tensor = torch.stack(batch)  # (N, M, 3, H, W)
    batch_tensor = batch_tensor.permute(0, 2, 1, 3, 4)  # (N, 3, M, 112, 112)

    features = model(batch_tensor)
    del batch_tensor
    torch.cuda.empty_cache()
    return features

@torch.no_grad()
def detect_objects_with_detr(key_frames, detr_model, clip_model, id2label):
    transform = transforms.ToTensor()
    features = []

    for frame in key_frames:
        frame_tensor = transform(Image.fromarray(frame)).unsqueeze(0).to(device)
        outputs = detr_model(frame_tensor)

        probs = outputs.logits.softmax(-1)[0, :, :-1]
        top_probs, top_labels = probs.max(-1)
        keep = top_probs > 0.8
        boxes = outputs.pred_boxes[0, keep].cpu()

        objs = []
        h, w, _ = frame.shape
        for box in boxes:
            x, y, w_box, h_box = box
            x_min = int((x - w_box / 2) * w)
            y_min = int((y - h_box / 2) * h)
            x_max = int((x + w_box / 2) * w)
            y_max = int((y + h_box / 2) * h)
            cropped = frame[y_min:y_max, x_min:x_max]
            if cropped.shape[0] >= 7 and cropped.shape[1] >= 7:
                objs.append(transform(Image.fromarray(cropped)).to(device))

        if objs:
            obj_features = extract_clip_features(objs, clip_model).mean(dim=0)
        else:
            obj_features = torch.zeros(512).to(device)
        features.append(obj_features)

        del frame_tensor, outputs
        torch.cuda.empty_cache()

    return torch.stack(features)

 
