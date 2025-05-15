import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.model import video_model
from dataset.dataset import ValVideoDataset
from utils.collate_fn import collate_val_fn
from utils.evaluate import evaluate_caption_scores, log_metrics_to_file
from utils.caption_utils import token_ids_to_text
from datasets import load_from_disk

def parse_args():
    parser = argparse.ArgumentParser(description="Video Captioning Validation Script")
    parser.add_argument('--test_data_path', type=str, default='../dataset_vatex/vatex0110/versions/1/vatex/json/public_test', help='Path to data path')
    parser.add_argument('--val_feature_dir', type=str, default='../video_features_vatex_MAE_v2', help='Path to feature directory')
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--val_checkpoint', type=str, default=None, help='Optional model checkpoint path')
    return parser.parse_args()

def evaluate_epoc(args):
    device = 'cuda'
    torch.cuda.empty_cache()

    # Load model weights nếu có
    if args.val_checkpoint:
        print(f"Loading checkpoint: {args.val_checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        video_model.load_state_dict(state_dict)
        video_model.to(device)

    

    # Load dataset
    dataset = load_from_disk(args.test_data_path)
    df = dataset.to_pandas()
    df = df.drop(columns=['chCap'])
    vatex_dataset = df.to_dict(orient='records')

    video_captions = {}
    for record in vatex_dataset:
        video_id = record['videoID']
        feature_file = os.path.join(args.val_feature_dir, f"{video_id}.pt")
        if os.path.exists(feature_file):
            video_captions[video_id] = record['enCap']
    
     

    if len(video_captions) == 0:
        print("No valid data found.")
        return

        
    val_data = [(vid, captions) for vid, captions in video_captions.items()]
    val_dataset = ValVideoDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=collate_val_fn)

    video_model.eval()
    predictions, references = [], []

    print("Running validation...")
    with torch.no_grad():
        for temporal, object_feats, context, caption_tokens in tqdm(val_loader):
            temporal = temporal.to(device)
            object_feats = object_feats.to(torch.float32).to(device)
            context = context.to(device)

            _, _, _, logits = video_model(temporal, object_feats, context, caption_tokens, mode="inference")
            for i in range(logits.shape[0]):
                pred = token_ids_to_text(logits[i], video_model.caption_head.tokenizer)
                truths = [token_ids_to_text(caption_token, video_model.caption_head.tokenizer) for caption_token in caption_tokens[i]]
                print(f"Pred: {''.join(pred)}, Truth: {''.join(truths[0])}")
                predictions.append(pred)
                references.append(truths)

    eval_result = evaluate_caption_scores(predictions, references)
    log_metrics_to_file(
        epoch=args.epoch,
        log_info=eval_result
    )

if __name__ == "__main__":
    args = parse_args()
    evaluate_epoc(args)
