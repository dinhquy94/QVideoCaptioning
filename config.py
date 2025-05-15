config = {
    "embed_dim": 1024,
    "proj_dims": {
        "temporal_in": 768,
        "object_in": 512,
        "context_in": 512,
        "out": 1024
    },
    "semantic_dim": 1024,
    "decoder_config": {
        "num_query_tokens": 15,
        "hidden_size": 1024,
        "proj_out_dim": 1280,
        "num_heads": 8,
        "decoder_model_path": "deepseek-ai/deepseek-vl2-tiny",
        "use_fp16": True,
        "num_beams": 5,
        "max_new_tokens": 32,
    },
    "object_head_config": {
        "embed_dim": 1024,
        "num_queries": 8,
        "num_heads": 8,
        "num_layers": 2,
        "proj_dim": 1024,
    },
    "action_head_config": {
        "embed_dim": 1024,
        "num_objects": 8,
        "num_actions": 8,
        "action_dim": 1024,
    },
    "global_head_config": {
        "embed_dim": 1024,
        "model_dim": 1024,
        "language_dim": 1024,
    }
}
