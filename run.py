"""
Module: run.py

Main entry point for training and evaluating video classification models on UCF50.
Supports:
  - Architecture switching: Factorized 3D CNN (R(2+1)D-18) vs. LRCN (2D ResNet-34 + BiLSTM)
  - Group-Aware Split saving & loading via splits.npy
  - Mixed-precision training and evaluation with 10-clip consensus
  - Weights & Biases experiment tracking
"""

import argparse
import glob
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.amp import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from models import LRCN, VideoResNet3D
from train import train
from utils import (
    compose_data_transforms,
    train_val_dloaders,
    transform_stats,
)
from video_datasets import VideoDataset, dataset_split, load_dataset


# Enable TF32 for Ampere Tensor Cores
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="UCF50 Action Recognition Benchmark")
    parser.add_argument("--frame_dir", type=str, default="data/UCF50_frames")
    parser.add_argument("--train_size", type=float, default=0.75)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--model_type", type=str, default="lrcn", choices=["lrcn", "3dcnn"])
    parser.add_argument("--cnn_backbone", type=str, default="resnet34")
    parser.add_argument("--n_classes", type=int, default=50)
    parser.add_argument("--fr_per_vid", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--eval_clips", type=int, default=10)
    return parser.parse_args()


def load_or_create_splits(samples, args):
    """Load cached splits or perform group-aware partitioning."""
    split_cache = "splits.npy"
    if os.path.exists(split_cache):
        print(f"Loading cached splits from {split_cache}...")
        splits = np.load(split_cache, allow_pickle=True).item()
        return splits["train"], splits["val"], splits["test"]

    print("Generating new group-aware splits...")
    tr_s, val_s, ts_s = dataset_split(
        samples, train_size=args.train_size, test_size=args.test_size
    )
    np.save(split_cache, {"train": tr_s, "val": val_s, "test": ts_s})
    return tr_s, val_s, ts_s


def build_model(args):
    """Instantiate model and configure parameter groups with discriminative learning rates."""
    if args.model_type == "3dcnn":
        model = VideoResNet3D(n_classes=args.n_classes, pretrained=True)
        # Higher capacity tuning: layer3=0.25x (~7.5e-5), layer4=0.5x (~1.5e-4), fc=1.0x (3e-4)
        param_groups = [
            {
                "params": [p for p in model.backbone.layer3.parameters() if p.requires_grad],
                "lr": args.learning_rate * 0.25
            },
            {
                "params": [p for p in model.backbone.layer4.parameters() if p.requires_grad],
                "lr": args.learning_rate * 0.5
            },
            {
                "params": [p for p in model.backbone.fc.parameters() if p.requires_grad],
                "lr": args.learning_rate
            }
        ]
    else:
        model = LRCN(
            hidden_size=256,
            n_layers=2,
            dropout_rate=0.3,
            n_classes=args.n_classes,
            pretrained=True,
            cnn_model=args.cnn_backbone
        )
        backbone_params = [
            p for n, p in model.base_model.named_parameters() if p.requires_grad
        ]
        head_params = [
            p for n, p in model.named_parameters()
            if not n.startswith("base_model") and p.requires_grad
        ]
        param_groups = [
            {"params": backbone_params, "lr": args.learning_rate * 0.1},
            {"params": head_params, "lr": args.learning_rate}
        ]

    return model, param_groups


def extract_temporal_clips(vid_dir, fr_per_vid, transform, model_type, num_clips=10):
    """Extract dense, uniformly spaced temporal clips across a video duration."""
    paths = sorted(glob.glob(os.path.join(vid_dir, "*.jpg")))
    batch_clips = []
    offsets = np.linspace(0.0, 1.0, num_clips)

    for off in offsets:
        clip_frames = []
        if len(paths) <= fr_per_vid:
            idxs = list(range(len(paths))) + [len(paths) - 1] * (fr_per_vid - len(paths))
        else:
            start = int(off * (len(paths) - fr_per_vid))
            idxs = list(range(start, start + fr_per_vid))

        for idx in idxs:
            img = Image.open(paths[idx]).convert("RGB")
            clip_frames.append(transform(img) if transform else img)

        stacked = torch.stack(clip_frames)
        if model_type == "3dcnn":
            stacked = stacked.permute(1, 0, 2, 3)
        batch_clips.append(stacked)

    return torch.stack(batch_clips)


def run_evaluation(args, model, test_samples, device):
    """Evaluate holdout test set using multi-clip temporal consensus."""
    _, ts_transforms = compose_data_transforms(*transform_stats(args.model_type))
    n_correct = 0
    total = len(test_samples)

    print(f"Evaluating {total} test videos with {args.eval_clips}-clip consensus...")
    model.eval()
    with torch.no_grad():
        for vid_dir, label in tqdm(test_samples):
            batch_clips = extract_temporal_clips(
                vid_dir,
                args.fr_per_vid,
                ts_transforms,
                args.model_type,
                num_clips=args.eval_clips
            ).to(device)

            with autocast("cuda", enabled=device.type == "cuda"):
                logits = model(batch_clips)
                probs = torch.softmax(logits, dim=1).mean(dim=0)
                pred = probs.argmax().item()

            if pred == int(label):
                n_correct += 1

    accuracy = 100.0 * n_correct / float(total)
    print(f"The overall test accuracy is {accuracy:.4f}%.")
    if wandb.run is not None:
        wandb.log({"test/consensus_accuracy": accuracy})


def run_training_pipeline(args, model, param_groups, splits_data, device):
    """Configure dataloaders, optimizers, and execute model training."""
    tr_tf, val_tf = compose_data_transforms(*transform_stats(args.model_type))

    dataloaders = train_val_dloaders(
        VideoDataset(splits_data[0], fr_per_vid=args.fr_per_vid, transforms=tr_tf),
        VideoDataset(splits_data[1], fr_per_vid=args.fr_per_vid, transforms=val_tf),
        args.batch_size,
        model=args.model_type
    )

    model.to(device)

    opt = torch.optim.AdamW(param_groups, weight_decay=1e-3)

    opt_setup = {
        "criterion": nn.CrossEntropyLoss(label_smoothing=0.1, reduction="sum"),
        "optimizer": opt,
        "scheduler": ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )
    }
    train_cfg = {
        "device": device,
        "save_dir": os.path.join("models", args.model_type),
        "n_epochs": args.n_epochs
    }

    train(dataloaders, model, opt_setup, train_cfg)


def main():
    """Main execution orchestrator."""
    args = parse_arguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_name = f"{args.model_type}-fpv{args.fr_per_vid}-bs{args.batch_size}"
    wandb.init(project="ucf50-video-action-recognition", name=run_name, config=vars(args))

    samples, _ = load_dataset(args.frame_dir)
    tr_s, val_s, ts_s = load_or_create_splits(samples, args)
    model, param_groups = build_model(args)

    save_dir = os.path.join("models", args.model_type)
    os.makedirs(save_dir, exist_ok=True)

    if args.mode == "train":
        run_training_pipeline(args, model, param_groups, (tr_s, val_s), device)
    elif args.mode == "eval":
        ckpt_path = args.ckpt if args.ckpt else os.path.join(save_dir, "best_model_wts.pt")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        run_evaluation(args, model, ts_s, device)


if __name__ == "__main__":
    main()
