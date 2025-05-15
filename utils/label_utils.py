import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
# Load SpaCy để trích xuất danh từ
nlp = spacy.load("en_core_web_sm")

# Load SBERT để tạo embedding
sbert_model = SentenceTransformer("sentence-transformers/all-roberta-large-v1")

device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = sbert_model.to(device)

def extract_noun_embeddings(caption):
    """Trích xuất danh từ từ caption và tạo embedding."""
    doc = nlp(caption)
    objects = set()

    for chunk in doc.noun_chunks:
        # Bỏ qua đại từ chung như "someone", "something"
        if chunk.root.pos_ == "PRON" and chunk.root.lower_ in {"someone", "something", "anyone"}:
            continue
        # Thêm noun phrase
        objects.add(chunk.text.strip())

    # Ngoài ra, kiểm tra các tân ngữ trực tiếp (dobj)
    for token in doc:
        if token.dep_ in ("dobj", "pobj", "attr") and token.pos_ == "NOUN":
            objects.add(token.text.strip())
    # print(caption, ":", noun_phrases)
    objects = list(objects)
    if not objects:
        objects = ["object"]  # Tránh lỗi nếu không có danh từ nào
    
    embeddings = sbert_model.encode(objects)  # (num_objects, 1024)
    return embeddings

 

def extract_verb_embeddings(caption):
    """Trích xuất hành động đầy đủ từ caption và tạo embedding (chỉ trả về hành động đầu tiên)."""
    doc = nlp(caption)
    actions = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    if not actions:
        actions = ["action"]
        
    embeddings = sbert_model.encode(actions)  # Ví dụ: 1024 chiều
    
    # embeddings = np.mean(embeddings, axis=0)  # Gộp lại thành 1 vector
    return embeddings

