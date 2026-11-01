"""
Module: video_datasets.py

This module provides classes and functions for loading and processing video datasets,
splitting them into training, validation, and test sets, and preparing data for video
classification models. It includes a custom PyTorch Dataset for videos stored as directories
of frame images, functions to load the dataset from a directory structure, split the dataset
using stratified sampling, and custom collate functions for handling variable-length video sequences.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from torch.nn.utils.rnn import pad_sequence

from PIL import Image

import os
import glob
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import re
from sklearn.model_selection import GroupShuffleSplit


class VideoDataset(Dataset):
    """
    PyTorch Dataset class for loading video data from directories of frame images.
    
    Each video is represented as a directory containing JPEG images of its frames.
    The dataset is provided as a dictionary mapping each video directory path to its label.
    
    Args:
        vid_dataset (dict): Dictionary where keys are video directory paths and values are integer labels.
        fr_per_vid (int): Number of frames per video to load (images are taken in order).
        transforms (callable, optional): A function/transform to apply to each frame image (e.g., resizing, normalization).
    """
    def __init__(self, vid_dataset, fr_per_vid, transforms=None):
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.transforms = transforms

    def __len__(self):
        """Return the number of video samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Load frames from the video directory corresponding to the given index, apply transforms,
        and return the stacked tensor of frames along with its label.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (frames_tensor, label) where frames_tensor is a tensor of shape (T, C, H, W)
                   with T being the number of frames (up to fr_per_vid) and label is an integer.
        """
        # Get all JPEG frame paths from the video directory and select up to fr_per_vid frames
        fr_paths = glob.glob(self.dataset[idx][0] + '/*.jpg')
        fr_paths = fr_paths[:self.fpv]
        
        # Open images using PIL
        fr_imgs = [Image.open(fr_path) for fr_path in fr_paths]
        
        # Get the label associated with the video
        fr_label = self.dataset[idx][1]
        
        # Apply transforms to each frame if provided, else keep original images
        fr_imgs_trans = [self.transforms(fr_img) for fr_img in fr_imgs] if self.transforms else fr_imgs

        # Stack transformed images into a tensor if available
        if len(fr_imgs_trans) > 0:
            fr_imgs_trans = torch.stack(fr_imgs_trans)

        return fr_imgs_trans, fr_label


def load_dataset(frame_dir):
    """
    Load the full video dataset from the specified directory.
    
    Each subdirectory in frame_dir is assumed to correspond to a video category.
    The function builds a dictionary where keys are paths to video directories and
    values are integer labels corresponding to each category.
    
    Args:
        frame_dir (str): Path to the directory containing subdirectories for each video category.
    
    Returns:
        tuple: (vid_dataset, label_dict)
            - vid_dataset (dict): Dictionary mapping video directory paths to integer labels.
            - label_dict (dict): Dictionary mapping video category names to integer labels.
    """
    label_dict = {vid_cat: idx for idx, vid_cat in enumerate(sorted(os.listdir(frame_dir)))}
    vid_dataset = {}
    print('Loading video dataset....')
    for vid_cat in tqdm(sorted(os.listdir(frame_dir))):
        vid_cat_path = os.path.join(frame_dir, vid_cat)
        for vid in os.listdir(vid_cat_path):
            vid_path = os.path.join(vid_cat_path, vid)
            vid_dataset[vid_path] = label_dict[vid_cat]
    return vid_dataset, label_dict

def extract_group_id(path_str):
    """
    Extract a unique group ID (e.g. 'BaseballPitch_g01') from the video path
    to ensure no recording session leaks across splits.
    """
    clip_name = os.path.basename(os.path.normpath(path_str))
    category_name = os.path.basename(os.path.dirname(os.path.normpath(path_str)))
    match = re.search(r'_g(\d+)', clip_name)
    group_num = match.group(1) if match else 'g00'
    return f"{category_name}_{group_num}"


def dataset_split(vid_dataset, tr_ratio, ts_ratio, seed=0):
    """
    Split the dataset into training, validation, and test sets using group-aware splitting.
    Guarantees that no video group (g01-g25) appears across multiple splits.
    
    Args:
        vid_dataset (dict): Dictionary mapping video paths to labels.
        tr_ratio (float): Proportion of the data to use for training.
        ts_ratio (float): Proportion of the data to use for testing.
        seed (int, optional): Random seed for reproducibility. Default is 0.
    
    Returns:
        tuple: (tr_dataset, val_dataset, ts_dataset)
            - tr_dataset (list): List of (video_path, label) tuples for the training set.
            - val_dataset (list): List of (video_path, label) tuples for the validation set.
            - ts_dataset (list): List of (video_path, label) tuples for the test set.


    """
    vid_paths = np.array(list(vid_dataset.keys()))
    vid_labels = np.array(list(vid_dataset.values()))
    groups = np.array([extract_group_id(p) for p in vid_paths])

    print('Splitting train/validation/test datasets with Group-Aware Splitting (No Data Leak)....')

    # Step 1: Split off Test Set by group
    gss_test = GroupShuffleSplit(n_splits=1, test_size=ts_ratio, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(vid_paths, vid_labels, groups))

    ts_dataset = [(vid_paths[i], int(vid_labels[i])) for i in test_idx]

    tr_val_paths = vid_paths[train_val_idx]
    tr_val_labels = vid_labels[train_val_idx]
    tr_val_groups = groups[train_val_idx]

    # Step 2: Split remaining data into Train and Validation by group
    val_ratio = 1.0 - tr_ratio - ts_ratio
    val_weight = val_ratio / (tr_ratio + val_ratio)

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_weight, random_state=seed)
    tr_idx, val_idx = next(gss_val.split(tr_val_paths, tr_val_labels, tr_val_groups))

    tr_dataset = [(tr_val_paths[i], int(tr_val_labels[i])) for i in tr_idx]
    val_dataset = [(tr_val_paths[i], int(tr_val_labels[i])) for i in val_idx]

    return tr_dataset, val_dataset, ts_dataset

def collate_fn_r3d_18(batch):
    """
    Collate function for 3D CNN models (e.g., R3D-18).
    
    Assumes each sample in the batch is a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). This function filters out any samples
    with no frames, stacks the video frame tensors, transposes the tensor dimensions as needed,
    and stacks the labels.
    
    Args:
        batch (list): List of samples, each as (video_frames, label).
    
    Returns:
        tuple: (imgs_tensor, labels_tensor)
            - imgs_tensor (Tensor): Stacked video frames tensor with shape adjusted for R3D-18.
            - labels_tensor (Tensor): Tensor of labels.
    """
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs) > 0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs) > 0]
    imgs_tensor = torch.stack(imgs_batch)
    imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor, labels_tensor

def collate_fn_rnn(batch):
    """
    Collate function for RNN-based models.
    
    Handles variable-length video sequences by padding them to the length of the longest sequence
    in the batch. Each sample in the batch is expected to be a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). The function returns a padded tensor
    of video frames with shape (batch_size, max_T, C, H, W) and a tensor of labels.
    
    Args:
        batch (list): List of samples, each as (video_frames, label).
    
    Returns:
        tuple: (padded_imgs, labels_tensor)
            - padded_imgs (Tensor): Padded tensor of video frames.
            - labels_tensor (Tensor): Tensor of labels.
    """
    # Unzip the batch into image tensors and labels
    imgs_batch, label_batch = list(zip(*batch))
    
    # Filter out any samples that have no frames
    valid_samples = [(imgs, label) for imgs, label in zip(imgs_batch, label_batch) if len(imgs) > 0]
    if not valid_samples:
        return None, None
    imgs_batch, label_batch = zip(*valid_samples)
    
    # Pad the video frame tensors along the time dimension (T)
    # Resulting shape: (batch_size, max_T, C, H, W)
    padded_imgs = pad_sequence(imgs_batch, batch_first=True)
    
    # Convert labels to a tensor
    labels_tensor = torch.tensor(label_batch)
    
    return padded_imgs, labels_tensor
