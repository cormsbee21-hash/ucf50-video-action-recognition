"""
Preprocess UCF50 video clips into uniformly sampled JPEG frames.
"""

import os
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor
import cv2
from tqdm import tqdm


def extract_frames_from_video(video_path: str, output_clip_dir: str, num_frames: int = 16) -> None:
    """
    Uniformly sample and write `num_frames` from a video to `output_clip_dir`.
    """
    os.makedirs(output_clip_dir, exist_ok=True)

    # Check if already processed to allow resuming
    existing_frames = glob.glob(os.path.join(output_clip_dir, "*.jpg"))
    if len(existing_frames) >= num_frames:
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return

    # Calculate uniform indices across the clip
    step = max(1, total_frames // num_frames)
    target_indices = {min(i * step, total_frames - 1) for i in range(num_frames)}

    frame_idx = 0
    saved_count = 0
    while cap.isOpened() and saved_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_indices:
            frame_filename = os.path.join(output_clip_dir, f"frame_{saved_count:02d}.jpg")
            # Write standard JPEG
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        frame_idx += 1

    cap.release()


def process_clip(args_tuple):
    video_path, output_clip_dir, num_frames = args_tuple
    extract_frames_from_video(video_path, output_clip_dir, num_frames)


def main():
    parser = argparse.ArgumentParser(description="Extract video frames for UCF50")
    parser.add_argument("--src_dir", type=str, default="data/UCF50", help="Root folder of raw UCF50 videos")
    parser.add_argument("--dest_dir", type=str, default="data/UCF50_frames", help="Destination folder for extracted frames")
    parser.add_argument("--fr_per_vid", type=int, default=16, help="Frames to extract per video")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes")
    args = parser.parse_args()

    # Find all .avi video files
    video_files = glob.glob(os.path.join(args.src_dir, "*", "*.avi"))
    print(f"Discovered {len(video_files)} video clips across UCF50.")

    tasks = []
    for vid_path in video_files:
        # Expected structure: data/UCF50/<Action_Class>/v_<Action>_gXX_cYY.avi
        category = os.path.basename(os.path.dirname(vid_path))
        clip_name = os.path.splitext(os.path.basename(vid_path))[0]
        output_dir = os.path.join(args.dest_dir, category, clip_name)
        tasks.append((vid_path, output_dir, args.fr_per_vid))

    print(f"Extracting {args.fr_per_vid} frames per clip into '{args.dest_dir}' using {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(executor.map(process_clip, tasks), total=len(tasks), desc="Processing Videos"))

    print("Preprocessing complete!")


if __name__ == "__main__":
    main()