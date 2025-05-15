from torch.nn.utils.rnn import pad_sequence

import torch
from torch.nn.utils.rnn import pad_sequence

def collate_action_fn(batch):
    """
    Hàm collate cho DataLoader, xử lý padding cho `label_embeddings`, `object_features`,
    và đảm bảo `action_embeddings` có dạng (batch_size, 1024).
    Cũng tạo mask cho `object_features` và `label_embeddings`.
    """
    temporal_features, object_features, context_features, label_embeddings, action_embeddings, caption_tokens, caption_embeds = zip(*batch)

    # Stack các đặc trưng theo batch
    temporal_features = torch.stack(temporal_features)  # (batch_size, seq_len, 512)
    context_features = torch.stack(context_features)    # (batch_size, seq_len, 512)

    # 🔹 Padding cho `object_features` (variable number of objects)
    padded_objects = pad_sequence([torch.tensor(obj, dtype=torch.float32) for obj in object_features],
                                  batch_first=True, padding_value=0)  # (batch_size, max_num_objects, feature_dim)


    # 🔹 Padding cho `label_embeddings`
    padded_labels = pad_sequence([torch.tensor(lbl, dtype=torch.float32) for lbl in label_embeddings], batch_first=True, padding_value=0)

    # 🔹 `action_embeddings`: (batch_size, 1024)
    # action_embeddings = torch.stack([torch.tensor(a, dtype=torch.float32) for a in action_embeddings])
    action_embeddings = pad_sequence([torch.tensor(action, dtype=torch.float32) for action in action_embeddings],
                                  batch_first=True, padding_value=0)  # (batch_size, max_num_objects, feature_dim)

    # 🔹 `caption_tokens` & `caption_embeds`
    caption_tokens = torch.stack([torch.tensor(a) for a in caption_tokens])
    caption_embeds = torch.stack([torch.tensor(a, dtype=torch.float32) for a in caption_embeds])

    return temporal_features, padded_objects, context_features, padded_labels, action_embeddings, caption_tokens, caption_embeds


def collate_val_fn(batch):
    """
    Hàm collate cho DataLoader, xử lý padding cho `object_features` và tạo mask cho `object_features`.
    """
    temporal_features, object_features, context_features, caption_tokens = zip(*batch)

    # 🔹 Stack các đặc trưng thời gian và ngữ cảnh (giả sử cùng shape)
    temporal_features = torch.stack(temporal_features)  # (batch_size, seq_len, 512)
    context_features = torch.stack(context_features)    # (batch_size, seq_len, 512)

    # 🔹 Padding `object_features` (num_objects khác nhau)
    padded_object_features = pad_sequence(
        [torch.tensor(obj, dtype=torch.float32) for obj in object_features],
        batch_first=True,
        padding_value=0
    )  # (batch_size, max_num_objects, feature_dim)
 
    # 🔹 Caption tokens (giữ nguyên list, bạn có thể pad nếu cần thêm xử lý sau)
    caption_tokens = [torch.tensor(a) for a in caption_tokens]
    # for a in caption_tokens:
    #     for b in a:
    #         print("b.shape", b.shape)

    return temporal_features, padded_object_features, context_features, caption_tokens

