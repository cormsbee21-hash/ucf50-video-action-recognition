"""
tests/test_dataset.py
"""
import pytest
from video_datasets import load_dataset, dataset_split, extract_group_id


def test_no_group_leakage_on_splits():
    """
    Ensure the set intersection of video groups across 
    train, val, and test splits is completely empty.
    """
    # 1. Mock dataset with 6 groups across 2 action classes
    mock_dataset = {
        "data/UCF50_frames/BaseballPitch/v_BaseballPitch_g01_c01": 0,
        "data/UCF50_frames/BaseballPitch/v_BaseballPitch_g01_c02": 0,
        "data/UCF50_frames/BaseballPitch/v_BaseballPitch_g02_c01": 0,
        "data/UCF50_frames/BaseballPitch/v_BaseballPitch_g02_c02": 0,
        "data/UCF50_frames/BaseballPitch/v_BaseballPitch_g03_c01": 0,
        "data/UCF50_frames/Basketball/v_Basketball_g01_c01": 1,
        "data/UCF50_frames/Basketball/v_Basketball_g01_c02": 1,
        "data/UCF50_frames/Basketball/v_Basketball_g02_c01": 1,
        "data/UCF50_frames/Basketball/v_Basketball_g03_c01": 1,
        "data/UCF50_frames/Basketball/v_Basketball_g04_c01": 1,
    }

    # 2. Run the split
    tr_dataset, val_dataset, ts_dataset = dataset_split(
        mock_dataset, tr_ratio=0.6, ts_ratio=0.2, seed=42
    )

    # 3. Extract unique group sets
    train_groups = {extract_group_id(path) for path, _ in tr_dataset}
    val_groups = {extract_group_id(path) for path, _ in val_dataset}
    test_groups = {extract_group_id(path) for path, _ in ts_dataset}

    # 4. Assert zero overlap between any pair
    train_test_overlap = train_groups.intersection(test_groups)
    train_val_overlap = train_groups.intersection(val_groups)
    val_test_overlap = val_groups.intersection(test_groups)

    assert len(train_test_overlap) == 0, f"Leak between Train & Test: {train_test_overlap}"
    assert len(train_val_overlap) == 0, f"Leak between Train & Val: {train_val_overlap}"
    assert len(val_test_overlap) == 0, f"Leak between Val & Test: {val_test_overlap}"