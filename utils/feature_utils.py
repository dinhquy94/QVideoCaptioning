import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
import random

import torch
import os

def load_video_features(video_id, feature_dir="../video_features_vatex_MAE_v2"):
    """
    Đọc file .pt của một video cụ thể.
    
    Args:
        video_id (str): ID của video.
        feature_dir (str): Thư mục chứa các file .pt.
        
    Returns:
        tuple: (temporal_features, object_features, context_features) hoặc None nếu file không tồn tại.
    """

    file_path = os.path.join(feature_dir, f"{video_id}.pt")
     
    if not os.path.exists(file_path):
        return None
      
    return torch.load(file_path)