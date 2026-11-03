"""
Module: preprocess.py

Extracts uniformly sampled frames from raw UCF50 video files (.avi) and stores
them as JPEG images organized by action class and video identity.
Supports multi-worker parallel execution via ProcessPoolExecutor with a clean progress bar.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import cv2
from tqdm import tqdm
from utils import get_frames, store_frames

# Prevent OpenCV internal threads from contending with ProcessPoolExecutor workers
cv2.setNumThreads(0)


def parse_arguments():
    """Parse command-line arguments for video preprocessing."""
    parser = argparse.ArgumentParser(
        description="Extract and store frames from UCF50 video dataset."
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        default="data/UCF50",
        help="Root path containing UCF50 raw action class video folders."
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        default="data/UCF50_frames",
        help="Destination directory to save extracted frame sequences."
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=32,
        help="Number of frames to extract per video (default: 32)."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel worker processes (default: 8)."
    )
    return parser.parse_args()


def process_video_task(task_args):
    """
    Worker task to extract and write frames for a single video.

    Args:
        task_args (tuple): (vid_path, out_dir, n_frames)
    """
    vid_path, out_dir, n_frames = task_args
    os.makedirs(out_dir, exist_ok=True)

    existing_frames = [f for f in os.listdir(out_dir) if f.endswith(".jpg")]
    if len(existing_frames) >= n_frames:
        return

    frames, _ = get_frames(vid_path, n_frames=n_frames)
    if frames:
        store_frames(frames, out_dir)


def preprocess_dataset(src_dir, dest_dir, n_frames=32, num_workers=8):
    """
    Traverse action category directories and process all video files concurrently.

    Args:
        src_dir (str): Source dataset directory containing class subfolders.
        dest_dir (str): Destination root directory for extracted frames.
        n_frames (int): Target frame count per video.
        num_workers (int): Number of parallel processes.
    """
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source video directory not found: {src_dir}")

    classes = sorted(entry.name for entry in os.scandir(src_dir) if entry.is_dir())
    print(f"Discovered {len(classes)} action categories in {src_dir}.")
    print(f"Extracting {n_frames} frames/video to {dest_dir} across {num_workers} workers...")

    tasks = []
    for cls_name in classes:
        cls_src_path = os.path.join(src_dir, cls_name)
        cls_dest_path = os.path.join(dest_dir, cls_name)

        vid_files = sorted(
            f for f in os.listdir(cls_src_path)
            if f.lower().endswith((".avi", ".mp4", ".mkv"))
        )
        for vid_file in vid_files:
            vid_base = os.path.splitext(vid_file)[0]
            vid_path = os.path.join(cls_src_path, vid_file)
            vid_out_dir = os.path.join(cls_dest_path, vid_base)
            tasks.append((vid_path, vid_out_dir, n_frames))

    print(f"Total videos queued: {len(tasks)}")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for _ in tqdm(
            executor.map(process_video_task, tasks),
            total=len(tasks),
            desc="Extracting UCF50 Frames",
            unit="vid",
            dynamic_ncols=True
        ):
            pass


def main():
    """CLI orchestrator for dataset preprocessing."""
    args = parse_arguments()
    preprocess_dataset(
        args.src_dir,
        args.dest_dir,
        n_frames=args.n_frames,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()