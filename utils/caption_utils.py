from models.model import video_model

def logits_to_text(logits):
    tokenizer = video_model.get_tokenizer()
    token_ids = torch.argmax(logits, dim=-1)  # (B, T)
    texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in token_ids]
    return texts


def token_ids_to_text(token_ids, tokenizer): 
    tokenizer = video_model.get_tokenizer()
    texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in token_ids]
    return texts

# Hàm mã hóa text thành token ids với padding và max_length
def get_caption_tokens(caption_text, max_length=30):
    tokenizer = video_model.get_tokenizer()
    return tokenizer(caption_text, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt").input_ids[0].tolist()

  