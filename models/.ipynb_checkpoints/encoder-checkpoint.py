import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn

class ObjectHead(nn.Module):
    def __init__(self, config):
        super(ObjectHead, self).__init__()
        self.embed_dim = config["embed_dim"]
        self.num_queries = config["num_queries"]

        self.query_embed = nn.Parameter(torch.randn(self.num_queries, self.embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config["num_heads"],
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["num_layers"])

        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=config["num_heads"],
            batch_first=True
        )

        self.projector = nn.Linear(self.embed_dim, config["proj_dim"])

    def forward(self, motion_features, object_features, context_features, object_mask=None):
        B, M, _ = object_features.shape
        N = motion_features.size(1)

        if object_mask is None:
            object_mask = (object_features.abs().sum(dim=-1) == 0)

        false_pad = torch.zeros((B, N), dtype=torch.bool, device=object_features.device)
        total_mask = torch.cat([false_pad, object_mask, false_pad], dim=1)

        combined = torch.cat([motion_features, object_features, context_features], dim=1)
        encoded = self.encoder(combined, src_key_padding_mask=total_mask)

        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        attended, _ = self.attn(query=queries, key=encoded, value=encoded, key_padding_mask=total_mask)
        projected = self.projector(attended)

        return attended, projected


class ActionHead(nn.Module):
    def __init__(self, config):
        super(ActionHead, self).__init__()
        self.embed_dim = config["embed_dim"]
        self.num_objects = config["num_objects"]
        self.num_actions = config["num_actions"]
        self.action_dim = config["action_dim"]

        self.W_alpha = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.U_alpha = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.b_alpha = nn.Parameter(torch.zeros(self.embed_dim))

        self.query_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.key_projection   = nn.Linear(self.embed_dim, self.embed_dim)
        self.value_projection = nn.Linear(self.embed_dim, self.embed_dim)

        self.fuse = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)

        self.fc = nn.Linear(self.embed_dim, self.action_dim)

    def forward(self, motion_features, object_features):
        B, N, D = motion_features.size()
        _, P, _ = object_features.size()

        M_proj = self.W_alpha(motion_features).unsqueeze(2)
        O_proj = self.U_alpha(object_features).unsqueeze(1)
        attn_scores = torch.tanh(M_proj + O_proj + self.b_alpha)
        attn_scores = attn_scores.mean(dim=-1)
        attn_weights = F.softmax(attn_scores, dim=-1)

        M_e = torch.bmm(attn_weights, object_features)
        M_combined = torch.cat([motion_features, M_e], dim=-1)
        M_combined = self.fuse(M_combined)
        M_combined = self.norm(M_combined)

        Q = self.query_projection(object_features)
        K = self.key_projection(M_combined)
        V = self.value_projection(M_combined)

        attn_output, _ = self.multi_head_attention(Q, K, V)
        projected_out = self.fc(attn_output)
        projected_out = F.normalize(projected_out, dim=-1)

        return attn_output, projected_out

    def multi_head_attention(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        return output, weights


class GlobalHead(nn.Module):
    def __init__(self, config):
        super(GlobalHead, self).__init__()
        self.embed_dim = config["embed_dim"]
        self.model_dim = config["model_dim"]
        self.language_dim = config["language_dim"]

        self.transformer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dropout=0.1,
            batch_first=True
        )
        self.fc_global = nn.Linear(self.embed_dim, self.model_dim)
        self.fc_language = nn.Linear(self.model_dim, self.language_dim)
        self.fc_output = nn.Linear(self.language_dim, 1)  # tùy task

    def forward(self, C, Ca, Ce):
        combined_context = torch.cat([C, Ce, Ca], dim=1)
        encoded_context = self.transformer(combined_context)

        global_representation = self.fc_global(encoded_context)
        language_representation = torch.mean(global_representation, dim=1)
        language_representation = self.fc_language(language_representation)

        return language_representation



 