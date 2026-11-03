"""
Module: video_datasets.py

This module provides classes and functions for loading and processing video datasets,
splitting them into training, validation, and test sets, and preparing data for video
classification models. It includes a custom PyTorch Dataset for videos stored as
directories of frame images, functions to load the dataset from a directory structure,
split the dataset using group-aware sampling, and custom collate functions for handling
variable-length video sequences.
"""

import glob
import os
import re
from PIL import Image
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm


class VideoDataset(Dataset):
    """
    PyTorch Dataset class for loading video data from directories of frame images.

    Each video is represented as a directory containing JPEG images of its frames.
    The dataset is provided as a dictionary mapping each video directory path to its label.

    Args:
        vid_dataset (dict): Dictionary mapping video directory paths to integer labels.
        fr_per_vid (int): Number of frames per video to load (images are taken in order).
        transforms (callable, optional): Transform to apply to each frame image
            (e.g., resizing, normalization).
    """

    def __init__(self, vid_dataset, fr_per_vid, transforms=None):
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path_to_vid, label = self.dataset[idx]
        frame_paths = sorted(glob.glob(os.path.join(path_to_vid, "*.jpg")))
        n_frames = len(frame_paths)

        if n_frames == 0:
            # Fallback for empty or corrupt directory: generate blank placeholder frames
            h, w = (224, 224)
            dummy_img = Image.new("RGB", (w, h), (0, 0, 0))
            if self.transforms:
                dummy_img = self.transforms(dummy_img)
            return torch.stack([dummy_img] * self.fpv), label

        if n_frames <= self.fpv:
            idxs = list(range(n_frames)) + [n_frames - 1] * (self.fpv - n_frames)
        else:
            idxs = np.linspace(0, n_frames - 1, self.fpv, dtype=int)

        frames = []
        for i in idxs:
            frame = Image.open(frame_paths[i]).convert("RGB")
            if self.transforms:
                frame = self.transforms(frame)
            frames.append(frame)

        frames = torch.stack(frames)
        return frames, label


def collate_fn_rnn(batch):
    """
    Collate function for RNN-based models (e.g., LRCN).
    Pads variable-length sequences to equal length.
    """
    videos, labels = zip(*batch)
    padded_videos = pad_sequence(videos, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_videos, labels


def collate_fn_r3d_18(batch):
    """
    Collate function for 3D CNN architectures (e.g., R3D-18).
    Permutes dimensions from (B, T, C, H, W) to (B, C, T, H, W).
    """
    videos, labels = zip(*batch)
    stacked_videos = torch.stack(videos, dim=0).permute(0, 2, 1, 3, 4)
    labels = torch.tensor(labels, dtype=torch.long)
    return stacked_videos, labels


def load_dataset(frame_dir):
    """
    Discovers all extracted video frame directories and associates them with integer labels.

    Args:
        frame_dir (str): Root directory containing category folders of video frames.

    Returns:
        tuple: (samples, class_to_idx)
    """
    classes = sorted(entry.name for entry in os.scandir(frame_dir) if entry.is_dir())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    samples = []
    print("Loading video dataset....")
    for cls_name in tqdm(classes):
        cls_dir = os.path.join(frame_dir, cls_name)
        for vid_name in os.listdir(cls_dir):
            vid_path = os.path.join(cls_dir, vid_name)
            if os.path.isdir(vid_path):
                samples.append((vid_path, class_to_idx[cls_name]))

    return samples, class_to_idx


def extract_group_id(vid_dir):
    """Extract group token from a video directory path."""
    match = re.search(r'g(\d+)', os.path.basename(vid_dir))
    return match.group(1) if match else "00"


def _extract_group_tags(samples):
    """Extract distinct group tags to prevent data leakage across splits."""
    groups = []
    for item in samples:
        vid_dir = item[0] if isinstance(item, (tuple, list)) else item
        groups.append(extract_group_id(vid_dir))
    return np.array(groups)


def _get_split_indices(groups, train_size, test_size, seed=42):
    """Compute split indices using GroupShuffleSplit without leaking groups."""
    indices = np.arange(len(groups))

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(indices, groups=groups))

    rem_size = train_size + (1.0 - train_size - test_size)
    val_ratio = (1.0 - train_size - test_size) / rem_size
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tr_rel, val_rel = next(gss_val.split(train_val_idx, groups=groups[train_val_idx]))

    return train_val_idx[tr_rel], train_val_idx[val_rel], test_idx


def dataset_split(samples, train_size=0.75, test_size=0.15, **kwargs):
    """
    Performs group-aware train/val/test splitting to prevent actor or background leak.

    Supports list of tuples, dictionaries, and legacy kwargs (tr_ratio, ts_ratio, seed).
    """
    train_size = kwargs.get('tr_ratio', train_size)
    test_size = kwargs.get('ts_ratio', test_size)
    seed = kwargs.get('seed', 42)

    sample_list = list(samples.items()) if isinstance(samples, dict) else samples

    print("Splitting train/validation/test datasets with Group-Aware Splitting (No Data Leak)....")
    groups = _extract_group_tags(sample_list)
    tr_idx, val_idx, ts_idx = _get_split_indices(groups, train_size, test_size, seed=seed)

    return (
        [sample_list[i] for i in tr_idx],
        [sample_list[i] for i in val_idx],
        [sample_list[i] for i in ts_idx],
    )
