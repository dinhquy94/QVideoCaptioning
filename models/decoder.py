

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from transformers import AutoModelForCausalLM


class QFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_query_tokens = config["num_query_tokens"]
        hidden_size = config["hidden_size"]
        num_heads = config["num_heads"]

        self.query_tokens = nn.Parameter(torch.randn(1, self.num_query_tokens, hidden_size))
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.type_embedding = nn.Embedding(3, hidden_size)  # 0: obj, 1: action, 2: context

    def forward(self, obj_feat, action_feat, context_feat):
        B, P_obj = obj_feat.size(0), obj_feat.size(1)
        P_act = action_feat.size(1)
        context_feat = context_feat.unsqueeze(1)

        type_ids = torch.cat([
            torch.zeros((B, P_obj), dtype=torch.long, device=obj_feat.device),
            torch.ones((B, P_act), dtype=torch.long, device=action_feat.device),
            torch.full((B, 1), 2, dtype=torch.long, device=context_feat.device)
        ], dim=1)

        memory = torch.cat([obj_feat, action_feat, context_feat], dim=1)
        type_emb = self.type_embedding(type_ids)
        memory += type_emb

        query_tokens = self.query_tokens.expand(B, -1, -1)
        attn_output, _ = self.attn(query_tokens, memory, memory)

        return attn_output





class CaptionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # QFormer + projection layer
        self.qformer = QFormer(config)
        self.proj = nn.Linear(config["hidden_size"], config["proj_out_dim"])

        # Load pretrained model + tokenizer
        processor = DeepseekVLV2Processor.from_pretrained(config["decoder_model_path"])
        self.tokenizer = processor.tokenizer

        decoder = AutoModelForCausalLM.from_pretrained(config["decoder_model_path"], trust_remote_code=True)
        if config.get("use_fp16", False):
            decoder = decoder.to(torch.bfloat16)
        self.lm_decoder = decoder.eval().to(self.device)
        for p in self.lm_decoder.parameters():
            p.requires_grad = False

        self.to(self.device)

    def forward(self, obj_feat, action_feat, context_feat, context_raw=None, truth_caption=None,
                mode='training', inference_strategy='greedy',
                temperature=1.0, top_k=30, top_p=0.9):

        obj_feat = obj_feat.to(self.device)
        action_feat = action_feat.to(self.device)
        context_feat = context_feat.to(self.device)

        qformer_output = self.qformer(obj_feat, action_feat, context_feat)
        q_feat_proj = self.proj(qformer_output)
        encoder_attention_mask = torch.ones(q_feat_proj.shape[:2], dtype=torch.long).to(self.device)

        if mode == 'training':
            assert truth_caption is not None
            B, T = q_feat_proj.shape[:2]

            prompt_inputs = self.tokenizer([""] * B, return_tensors="pt", padding=True).to(self.device)
            prompt_embeds = self.lm_decoder.prepare_inputs_embeds(prompt_inputs["input_ids"])
            caption_embeds = self.lm_decoder.prepare_inputs_embeds(truth_caption.to(self.device))

            combined_embeds = torch.cat([
                q_feat_proj.to(torch.bfloat16).contiguous(),
                prompt_embeds,
                caption_embeds
            ], dim=1)

            attention_mask = torch.cat([
                torch.ones((B, T), dtype=torch.long).to(self.device),
                prompt_inputs["attention_mask"].to(self.device),
                torch.ones((B, caption_embeds.shape[1]), dtype=torch.long).to(self.device)
            ], dim=1).to(self.device)

            labels = torch.cat([
                torch.full((B, T + prompt_embeds.shape[1]), -100, dtype=torch.long).to(self.device),
                truth_caption.to(self.device)
            ], dim=1).to(self.device)

            outputs = self.lm_decoder(
                attention_mask=attention_mask,
                inputs_embeds=combined_embeds,
                labels=labels,
                return_dict=True,
                use_cache=True,
            )
            return outputs.loss, outputs.logits

        elif mode == 'inference':
            B, T = q_feat_proj.shape[:2]
            prompt_inputs = self.tokenizer([""] * B, return_tensors="pt", padding=True).to(self.device)
            prompt_embeds = self.lm_decoder.prepare_inputs_embeds(prompt_inputs["input_ids"])

            combined_embeds = torch.cat([q_feat_proj.to(torch.bfloat16), prompt_embeds], dim=1)
            attention_mask = torch.cat([
                torch.ones((B, T), dtype=torch.long).to(self.device),
                prompt_inputs["attention_mask"].to(self.device)
            ], dim=1).to(self.device)

            generated = self.lm_decoder.generate(
                attention_mask=attention_mask,
                inputs_embeds=combined_embeds,
                use_cache=True,
                do_sample=(inference_strategy == 'sampling'),
                num_beams=self.config.get("num_beams", 5),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.config.get("max_new_tokens", 32),
                early_stopping=True,
            )
            return generated

        else:
            raise ValueError(f"Unknown mode: {mode}")

 