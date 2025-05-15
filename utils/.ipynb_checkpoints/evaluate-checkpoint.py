from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import importlib 
import os
from datetime import datetime

def evaluate_caption_scores(predictions, references):
    smoothie = SmoothingFunction().method4

    # Làm sạch: nối các token lại thành câu
    pred_sentences = [' '.join([t for t in pred if t.strip()]) for pred in predictions]
    
    # Các references có thể có nhiều caption cho mỗi video
    ref_sentences = {
        i: [' '.join([t for t in ref if t.strip()]) for ref in refs]
        for i, refs in enumerate(references)
    }

    # Khởi tạo điểm BLEU theo từng cấp độ
    bleu_scores = {
        1: [],
        2: [],
        3: [],
        4: []
    }

    for i, pred in enumerate(pred_sentences):
        pred_tokens = pred.split()
        ref_tokens = [ref.split() for ref in ref_sentences[i]]

        # Tính BLEU-1 đến BLEU-4
        bleu_scores[1].append(sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie))
        bleu_scores[2].append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
        bleu_scores[3].append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.34, 0), smoothing_function=smoothie))
        bleu_scores[4].append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))

    # Trung bình mỗi loại BLEU
    bleu1 = sum(bleu_scores[1]) / len(bleu_scores[1])
    bleu2 = sum(bleu_scores[2]) / len(bleu_scores[2])
    bleu3 = sum(bleu_scores[3]) / len(bleu_scores[3])
    bleu4 = sum(bleu_scores[4]) / len(bleu_scores[4])
    bleu_avg = (bleu1 + bleu2 + bleu3 + bleu4) / 4

    # CIDEr Score (nếu có thư viện)
    cider_score = None
    if importlib.util.find_spec("pycocoevalcap"):
        from pycocoevalcap.cider.cider import Cider

        gts = {i: ref_sentences[i] for i in range(len(ref_sentences))}
        res = {i: [pred_sentences[i]] for i in range(len(pred_sentences))}
        scorer = Cider()
        cider_score, _ = scorer.compute_score(gts, res)
    else:
        print("⚠️  CIDEr evaluation skipped (pycocoevalcap not installed)")

    return {
        "BLEU-1": bleu1,
        "BLEU-2": bleu2,
        "BLEU-3": bleu3,
        "BLEU-4": bleu4,
        "BLEU-avg": bleu_avg,
        "CIDEr": cider_score
    }



def log_metrics_to_file(epoch, log_info, log_dir="logs"):
    """
    Ghi log đánh giá vào file theo ngày, kèm thời gian cụ thể.

    Args:
        epoch (int): Epoch hiện tại.
        log_info (dict): Các giá trị loss cần log.
        log_dir (str): Thư mục lưu log.
    """
    # Lấy ngày hiện tại để đặt tên file
    today_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    log_path = os.path.join(log_dir, f"{today_str}.log")

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(log_dir, exist_ok=True)

    # Ghi nội dung log
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{time_str}] Epoch {epoch + 1}\n") 
        for k, v in log_info.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("-" * 40 + "\n")

