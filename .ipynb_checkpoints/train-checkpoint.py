import argparse
import os
import torch
import random
import json
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
from models.model import video_model
from dataset.dataset import VideoDataset
from utils.collate_fn import collate_action_fn
from loss import cosine_loss, hungarian_loss
from validate import evaluate_epoc

def parse_args():
    parser = argparse.ArgumentParser(description="Video Captioning Training Script")
    parser.add_argument('--dataset_path', type=str, default='../dataset_vatex/vatex0110/versions/1/vatex/json/train', help='Path to dataset')
    parser.add_argument('--test_data_path', type=str, default='../dataset_vatex/vatex0110/versions/1/vatex/json/public_test', help='Path to data path')
    parser.add_argument('--feature_dir', type=str, default='../video_features_vatex_MAE_v2', help='Path to video features')
    parser.add_argument('--val_feature_dir', type=str, default='../video_features_vatex_MAE_v2', help='Path to feature directory')
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--checkpoint', type=str, default='../model_deepseek_epoc2.pt', help='Path to pretrained model')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Where to save checkpoints')
    return parser.parse_args()

def main(args):
    torch.cuda.empty_cache()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model weights
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    video_model.load_state_dict(state_dict)
    video_model.to('cuda')

    # Load dataset
    dataset = load_from_disk(args.dataset_path)
    df = dataset.to_pandas()
    df = df.drop(columns=['chCap'])
    vatex_dataset = df.to_dict(orient='records')

    video_captions = {}
    for record in vatex_dataset:
        video_id = record['videoID']
        feature_file = os.path.join(args.feature_dir, f"{video_id}.pt")
        if os.path.exists(feature_file):
            video_captions[video_id] = record['enCap']

    all_video_ids = list(video_captions.keys())
    random.shuffle(all_video_ids)

    train_data = []
    for video_id in all_video_ids:
        for caption in video_captions[video_id]:
            train_data.append((caption, video_id))

    train_dataset = VideoDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_action_fn)

    optimizer = torch.optim.Adam(video_model.parameters(), lr=args.lr)
    device = 'cuda'

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(args.num_epochs):
        video_model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=True)

        for i, (temporal, object_feats, context, label_emb, action_emb, caption_tokens, caption_embeds) in enumerate(progress_bar):
            temporal = temporal.to(device)
            object_feats = object_feats.to(torch.float32).to(device)
            context = context.to(device)
            label_emb = label_emb.to(device)
            action_emb = action_emb.to(device)
            caption_tokens = caption_tokens.to(device)
            caption_embeds = caption_embeds.to(device)

            obj_out, act_out, glob_out, cap_logits, cap_loss = video_model(
                temporal, object_feats, context, caption_tokens, mode="training"
            )

            action_loss = hungarian_loss(act_out, action_emb)
            object_loss = hungarian_loss(obj_out, label_emb, 'cosine') 
            global_loss_val = cosine_loss(glob_out, caption_embeds)

            loss = 0.2 * action_loss + 0.2 * object_loss + 0.4 * cap_loss + 0.2 * global_loss_val

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss

            progress_bar.set_postfix({
                "Batch Loss": batch_loss, 
                "Loss": f"caption: {cap_loss.item():.3f}, object: {object_loss.item():.3f}, global: {global_loss_val.item():.3f}, action: {action_loss.item():.3f}",
            })
       
        checkpoint_path = os.path.join(args.output_dir, f"model_checkpoint-epoch-{epoch}.pt")
        args.val_checkpoint = checkpoint_path
        args.epoch = epoch
        torch.save(video_model.state_dict(), checkpoint_path)
        evaluate_epoc(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
