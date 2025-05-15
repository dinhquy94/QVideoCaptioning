import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
import random

import torch
import os
from utils.label_utils import extract_noun_embeddings, extract_verb_embeddings, sbert_model
from utils.caption_utils import get_caption_tokens
 

from utils.feature_utils import load_video_features

class VideoDataset(Dataset):
    def __init__(self, video_data):
        """
        video_data: List chứa các video dưới dạng [(caption, video_id), ...]
        """
        self.video_data = video_data

    def preprocess(self, video_id):
        """Hàm này trả về temporal_features, object_features, context_features.""" 
        temporal_features, object_features, context_features = load_video_features(video_id)
        return temporal_features, object_features, context_features

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        caption, video_id = self.video_data[idx]
        temporal_features, object_features, context_features = self.preprocess(video_id)

        # Trích xuất danh từ và tạo embedding
        label_embeddings = extract_noun_embeddings(caption)
        action_feature = extract_verb_embeddings(caption)
        caption_tokens = get_caption_tokens(caption)
        caption_embed = sbert_model.encode(caption)

        return temporal_features, object_features, context_features, torch.tensor(label_embeddings), action_feature, caption_tokens, caption_embed




class ValVideoDataset(Dataset):
    def __init__(self, video_data):
        """
        video_data: List chứa các video dưới dạng [(video_id, [caption1, caption2, ..., caption5]), ...]
        """
        self.video_data = video_data

    def preprocess(self, video_id):
        """Hàm này trả về temporal_features, object_features, context_features."""
        temporal_features, object_features, context_features = load_video_features(video_id)
        return temporal_features, object_features, context_features

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        video_id, captions = self.video_data[idx]
        temporal_features, object_features, context_features = self.preprocess(video_id)
        caption_tokens = []
        # Duyệt qua tất cả các caption và trích xuất thông tin
        for caption in captions: 
            caption_token = get_caption_tokens(caption)
            caption_tokens.append(caption_token) 
            
        # Return: trả về nhiều caption để tính BLEU, CIDEr
        return temporal_features, object_features, context_features, caption_tokens


