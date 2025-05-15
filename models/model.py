import torch
import torch.nn as nn
from models.encoder import ObjectHead, GlobalHead, ActionHead
from models.decoder import CaptionHead
from transformers import AutoModelForCausalLM
from config import config


class VidCapModel(nn.Module):
    def __init__(self, config):
        super(VidCapModel, self).__init__()
        self.config = config

        self.object_head = ObjectHead(config["object_head_config"])
        self.action_head = ActionHead(config["action_head_config"])
        self.global_head = GlobalHead(config["global_head_config"])
        self.caption_head = CaptionHead(config["decoder_config"])

        proj = config["proj_dims"]
        self.temporal_proj = nn.Linear(proj["temporal_in"], proj["out"])
        self.object_proj   = nn.Linear(proj["object_in"], proj["out"])
        self.context_proj  = nn.Linear(proj["context_in"], proj["out"])

        self.semantic_mapper = nn.Linear(config["embed_dim"], config["semantic_dim"])

    def get_tokenizer(self):
        return self.caption_head.tokenizer

    def forward(self, temporal_features, object_features, context_features, caption_tokens, mode='training'):
        C_project = self.context_proj(context_features)
        M_project = self.temporal_proj(temporal_features)
        object_project = self.object_proj(object_features)

        ξ, object_head_output = self.object_head(M_project, object_project, C_project)
        attn_action, action_head_output = self.action_head(M_project, ξ)
        global_head_output = self.global_head(M_project, attn_action, ξ)

        if mode == 'training':
            caption_loss, captions_logits = self.caption_head(
                object_head_output,
                action_head_output,
                global_head_output,
                C_project,
                caption_tokens,
                mode='training',
                top_k=0
            )
            return object_head_output, action_head_output, global_head_output, captions_logits, caption_loss

        elif mode == 'inference':
            captions_logits = self.caption_head(
                object_head_output,
                action_head_output,
                global_head_output,
                C_project,
                caption_tokens,
                mode='inference',
                inference_strategy='beam'
            )
            return object_head_output, action_head_output, global_head_output, captions_logits

        else:
            raise ValueError(f"Unsupported mode: {mode}")

video_model = VidCapModel(config).to('cuda')
