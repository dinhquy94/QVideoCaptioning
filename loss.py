import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# Hàm loss Euclidean
def euclidean_loss(pred, target):
    return torch.norm(pred - target, p=2, dim=-1).mean()


# Hàm loss Cosine
def cosine_loss(pred, target):
    cos_sim = F.cosine_similarity(pred, target, dim=-1)  # Tính Cosine similarity
    return 1 - cos_sim.mean()  # Cosine loss, minimize similarity = maximize cosine distance
 


def hungarian_loss(object_head_output, label_emb, loss_type='cosine'):
    B, N, D = object_head_output.shape
    total_loss = 0.0

    for b in range(B):
        pred = object_head_output[b]     # (N, D)
        target = label_emb[b]            # (M, D)

        # Lọc ra các vector label không phải padding (toàn 0)
        valid_mask = (target.abs().sum(dim=1) != 0)  # (M,)
        target = target[valid_mask]  # (M_valid, D)
        
        if target.shape[0] == 0:
            # Nếu không có label hợp lệ, bỏ qua sample này
            continue

        if loss_type == 'l1':
            cost_matrix = torch.cdist(pred, target, p=1)  # (N, M_valid)
        elif loss_type == 'cosine':
            pred_norm = F.normalize(pred, p=2, dim=1)
            target_norm = F.normalize(target, p=2, dim=1)
            cost_matrix = 1 - torch.matmul(pred_norm, target_norm.T)  # (N, M_valid)
        else:
            raise ValueError("Unsupported loss_type. Choose 'l1' or 'cosine'.")

        row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

        matched_pred = pred[row_ind]
        matched_target = target[col_ind]

        if loss_type == 'l1':
            loss = F.l1_loss(matched_pred, matched_target, reduction='mean')
        elif loss_type == 'cosine':
            loss = 1 - F.cosine_similarity(matched_pred, matched_target, dim=1).mean()

        total_loss += loss

    return total_loss / B
