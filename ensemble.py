"""
Module: ensemble.py

Performs soft-voting ensemble evaluation between LRCN and Factorized 3D-CNN (R(2+1)D-18)
on the UCF50 holdout test set using 10-clip temporal consensus.
Logs final ensemble accuracy to Weights & Biases.
"""

import argparse
import os
import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm
import wandb

from models import LRCN, VideoResNet3D
from run import extract_temporal_clips
from utils import compose_data_transforms, transform_stats


def parse_args():
    """Parse command-line arguments for ensemble evaluation."""
    parser = argparse.ArgumentParser(description="UCF50 Dual Model Ensemble Evaluation")
    parser.add_argument("--splits_path", type=str, default="splits.npy",
                        help="Path to saved dataset splits file.")
    parser.add_argument("--lrcn_ckpt", type=str,
                        default="models/lrcn/best_model_wts.pt",
                        help="Path to trained LRCN checkpoint.")
    parser.add_argument("--r3d_ckpt", type=str,
                        default="models/3dcnn/best_model_wts.pt",
                        help="Path to trained 3D-CNN checkpoint.")
    parser.add_argument("--fr_per_vid", type=int, default=16,
                        help="Frames sampled per clip window.")
    parser.add_argument("--eval_clips", type=int, default=10,
                        help="Number of temporal consensus clips per video.")
    parser.add_argument("--lrcn_weight", type=float, default=0.35,
                        help="Soft-voting weight for LRCN probabilities.")
    parser.add_argument("--r3d_weight", type=float, default=0.65,
                        help="Soft-voting weight for 3D-CNN probabilities.")
    return parser.parse_args()


def load_models(args, device):
    """Instantiate and load weights for LRCN and R(2+1)D-18."""
    print("Loading LRCN checkpoint from:", args.lrcn_ckpt)
    lrcn = LRCN(hidden_size=256, n_layers=2, dropout_rate=0.3, n_classes=50, cnn_model="resnet34")
    lrcn.load_state_dict(torch.load(args.lrcn_ckpt, map_location=device))
    lrcn.to(device).eval()

    print("Loading 3D-CNN checkpoint from:", args.r3d_ckpt)
    r3d = VideoResNet3D(n_classes=50)
    r3d.load_state_dict(torch.load(args.r3d_ckpt, map_location=device))
    r3d.to(device).eval()

    return lrcn, r3d


def main():
    """Main execution function for the ensemble evaluation pipeline."""
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="ucf50-video-action-recognition",
        name="ensemble-soft-voting-consensus",
        config=vars(args)
    )

    if not os.path.exists(args.splits_path):
        raise FileNotFoundError(f"Splits file not found at {args.splits_path}")

    splits = np.load(args.splits_path, allow_pickle=True).item()
    test_samples = splits["test"]
    total = len(test_samples)
    print(f"Loaded {total} holdout test samples from {args.splits_path}.")

    lrcn_model, r3d_model = load_models(args, device)

    # Load data transforms
    _, lrcn_tf = compose_data_transforms(*transform_stats("lrcn"))
    _, r3d_tf = compose_data_transforms(*transform_stats("3dcnn"))

    correct = 0
    print(f"Evaluating ensemble using {args.eval_clips}-clip consensus (Weights: LRCN={args.lrcn_weight}, 3D-CNN={args.r3d_weight})...")

    with torch.no_grad():
        for vid_dir, label in tqdm(test_samples):
            # Extract 10-clip temporal batches
            lrcn_clips = extract_temporal_clips(
                vid_dir, args.fr_per_vid, lrcn_tf, "lrcn", num_clips=args.eval_clips
            ).to(device)

            r3d_clips = extract_temporal_clips(
                vid_dir, args.fr_per_vid, r3d_tf, "3dcnn", num_clips=args.eval_clips
            ).to(device)

            with autocast("cuda", enabled=device.type == "cuda"):
                # Average probabilities over temporal clips
                p_lrcn = torch.softmax(lrcn_model(lrcn_clips), dim=1).mean(dim=0)
                p_r3d = torch.softmax(r3d_model(r3d_clips), dim=1).mean(dim=0)

                # Soft voting blend
                ensemble_probs = (args.lrcn_weight * p_lrcn) + (args.r3d_weight * p_r3d)
                pred = ensemble_probs.argmax().item()

            if pred == int(label):
                correct += 1

    accuracy = 100.0 * correct / float(total)

    print("\n" + "=" * 60)
    print(f"Final Test Accuracy (Soft-Voting Ensemble): {accuracy:.4f}%")
    print("=" * 60)

    if wandb.run is not None:
        wandb.log({"test/ensemble_accuracy": accuracy})
        wandb.finish()


if __name__ == "__main__":
    main()
