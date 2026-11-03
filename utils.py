"""
Module: utils.py

This module provides helper functions for video processing and data transformations
for video classification tasks. It includes functions for:
    - Uniformly sampling frames from videos.
    - Storing extracted frames as JPEG images.
    - Retrieving image transformation statistics based on the model type.
    - Composing data transforms for training and validation/test datasets.
    - Creating DataLoaders for training, validation, and testing, using custom collate functions.
"""

# pylint: disable=no-member

import os
import cv2
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from video_datasets import collate_fn_r3d_18, collate_fn_rnn


def get_frames(vid, n_frames=1):
    """
    Uniformly sample frames from a video file.

    Args:
        vid (str): Path to the video file.
        n_frames (int): Number of frames to sample from the video.

    Returns:
        tuple: (frames, v_len)
            - frames (list): List of sampled frames (as numpy arrays in RGB format).
            - v_len (int): Total number of frames in the video.

    Notes:
        - If the video cannot be opened or contains no frames, an empty list and 0 are returned.
        - Frames are sampled at uniformly spaced indices.
    """
    frames = []
    v_cap = cv2.VideoCapture(vid)
    if not v_cap.isOpened():
        print("Failed to open video:", vid)
        return frames, 0
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if v_len <= 0:
        print("No frames found in video:", vid)
        v_cap.release()
        return frames, 0
    frame_idx = np.linspace(0, v_len - 1, n_frames + 1, dtype=np.int16)
    for idx in range(v_len):
        success, frame = v_cap.read()
        if not success:
            continue
        if idx in frame_idx:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    v_cap.release()
    return frames, v_len


def store_frames(frames, store_path):
    """
    Save a list of frames as JPEG images to the specified directory.

    Each frame is converted from RGB to BGR format (as expected by OpenCV)
    before saving.

    Args:
        frames (list): List of frames (numpy arrays in RGB format) to save.
        store_path (str): Directory path where the frames will be stored.
    """
    for idx, frame in enumerate(frames):
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path_to_frame = os.path.join(store_path, f"frame{idx}.jpg")
        cv2.imwrite(path_to_frame, bgr_frame)


def transform_stats(model='lrcn'):
    """
    Retrieve transformation statistics based on the model type.

    For 'lrcn', images are resized to 224x224; for '3dcnn', images are resized to 112x112.
    Also returns the mean and standard deviation values used for normalization.

    Args:
        model (str): Type of model ('lrcn' or '3dcnn').

    Returns:
        tuple: (h, w, mean, std)
    """
    if model == 'lrcn':
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model in ['3dcnn', 'r3d_18']:
        h, w = 112, 112
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
    else:
        raise ValueError('model_type arg is undefined....')
    return h, w, mean, std


def compose_data_transforms(height, width, mean, std):
    """
    Compose and return data transforms for training and validation/test datasets.

    Training transforms introduce:
    - RandomResizedCrop (scale 0.8 to 1.0) to simulate camera distance changes.
    - ColorJitter (brightness, contrast) to make model invariant to lighting differences.
    - Consistent spatial normalization.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((int(height * 1.14), int(width * 1.14))),
        transforms.RandomCrop((height, width)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transforms, val_test_transforms


def train_val_dloaders(train_dataset, val_dataset, batch_size, model='lrcn'):
    """
    Create DataLoaders for training and validation datasets.

    Selects the appropriate collate function based on the model type.
    """
    collate = collate_fn_rnn if model == "lrcn" else collate_fn_r3d_18
    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=4,              # Drop from 8 -> 2 to prevent /dev/shm pool crashes
        pin_memory=True,
        persistent_workers=False,   # Allows system to flush shared buffers
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=batch_size,      # Match batch_size instead of 2 * batch_size
        shuffle=False,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
    )
    return {'train': train_dl, 'val': val_dl}


def test_dloaders(test_dataset, batch_size, model='lrcn'):
    """
    Create a DataLoader for the test dataset.

    Selects the appropriate collate function based on the model type.
    """
    collate = collate_fn_rnn if model == "lrcn" else collate_fn_r3d_18
    test_dl = DataLoader(
        test_dataset, batch_size=2 * batch_size, shuffle=False,
        collate_fn=collate, num_workers=4, pin_memory=True
    )
    return {'test': test_dl}
