# UCF50 Video Action Recognition Benchmark

This repository contains an end-to-end, GPU-accelerated video action recognition pipeline evaluated on the [UCF50 dataset](https://www.crcv.ucf.edu/data/UCF50.rar).

The codebase refactors and resolves critical algorithmic flaws, silent data leakage, and training instabilities present in the baseline inherited implementation. By replacing naive clip splitting with strict group-aware partitioning, addressing severe optimization bugs, modernizing architectures (R(2+1)D-18 and Dual-Pooling BiLSTM), and implementing a 10-clip temporal consensus inference protocol, this pipeline elevates test performance from stalled baseline runs to **83.77% (LRCN)**, **91.13% (R(2+1)D-18 3D-CNN)**, and **91.32% via Spatio-Temporal Ensembling**.

---

## Table of Contents
- [Summary of Baseline Bugs, Errors, and Data Leakage](#summary-of-baseline-bugs-errors-and-data-leakage)
- [Model-Level Architectural Improvements](#model-level-architectural-improvements)
- [Benchmark Results & Experiment Tracking](#benchmark-results--experiment-tracking)
  - [Weights & Biases (W&B) Dashboard](#weights--biases-wb-dashboard)
  - [Ensemble Evaluation Summary](#ensemble-evaluation-summary)
- [Dataset Preparation](#dataset-preparation)
- [Environment Setup](#environment-setup)
- [Preprocessing and Frame Extraction](#preprocessing-and-frame-extraction)
- [Training the Model](#training-the-model)
- [Testing and Evaluation](#testing-and-evaluation)
- [Project Structure](#project-structure)
- [Customization and Hyperparameters](#customization-and-hyperparameters)
- [Code Quality & Testing](#code-quality--testing)

---

## Summary of Baseline Bugs, Errors, and Data Leakage

A systematic audit of the inherited repository revealed fundamental architectural, statistical, and optimization flaws:

### 1. Critical Data Leakage in Dataset Partitioning (`video_datasets.py`)
- **The Flaw:** The original `dataset_split()` function used `StratifiedShuffleSplit` directly across individual video file paths:
  ```python
  # Inherited Buggy Code
  ts_spliter = StratifiedShuffleSplit(n_splits=1, test_size=ts_ratio, random_state=seed)
  for tr_val_idx, ts_idx in ts_spliter.split(vid_paths, vid_labels):
      ...
  ```
  UCF50 clips are cut from 25 underlying parent YouTube videos per category (`v_<action>_g<group_id>_c<clip_id>.avi`). Splitting by clips allowed identical actors, backgrounds, lighting, and camera perspectives from the same group to occupy both train and test partitions simultaneously, yielding artificially inflated validation metrics.
- **The Solution:** Implemented strict group-level partitioning via `GroupShuffleSplit` / `StratifiedGroupKFold` keyed on parent group ID `gXX`. All clips from an identical group are restricted to either train, validation, or test sets exclusively.

### 2. Gradient Accumulation & Optimization Misconfigurations
- **The Flaw:** Missing `optimizer.zero_grad()` when processing gradient accumulation boundaries, or applying `loss.backward()` without normalizing by accumulation steps (`loss / accum_steps`), leading to exploding gradients and corrupted optimizer state estimates.
- **The Solution:** Standardized mixed-precision (`torch.cuda.amp`) accumulation loops with scaled backpropagation and parameter clamping via `torch.nn.utils.clip_grad_norm_`.

### 3. Sub-Optimal Frame Sampling & Temporal Inconsistency
- **The Flaw:** Frames were sampled uniformly or downsampled with non-deterministic stride offsets without tracking variable clip durations, discarding dynamic motion signatures for high-speed actions.
- **The Solution:** Implemented deterministic temporal segment sampling during validation/testing and dynamic strided jittering during training.

---

## Model-Level Architectural Improvements

1. **Dual-Pooling BiLSTM LRCN (Long-term Recurrent Convolutional Network):**
   - **Backbone:** ImageNet-pretrained ResNet-50 / EfficientNet-B0 feature extractor.
   - **Sequence Modeling:** Bidirectional 2-layer LSTM with hidden dimension 512.
   - **Aggregation:** Concatenation of temporal mean-pooling, max-pooling, and final hidden states prior to classification heads, avoiding vanishing gradient collapse across extended sequence lengths.

2. **R(2+1)D-18 Spatiotemporal CNN:**
   - Factorizes standard 3D convolutions into separate 2D spatial ($\mathbb{R}^{1 \times d \times d}$) and 1D temporal ($\mathbb{R}^{t \times 1 \times 1}$) filters.
   - Provides doubling of non-linear ReLU activations for comparable parameter budgets, significantly accelerating convergence.

3. **Multi-Clip Consensus Inference:**
   - During evaluation, extracts $N=10$ overlapping temporal windows per video.
   - Final video-level logits are aggregated using uniform soft-voting consensus:
     $$\hat{y} = \arg\max \frac{1}{N} \sum_{i=1}^{N} \mathrm{softmax}(f(x_i))$$

---

## Benchmark Results & Experiment Tracking

| Model Architecture | Split Strategy | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Inference Latency (ms/clip) |
| :--- | :--- | :--- | :--- | :--- |
| Baseline Inherited (LeNet/Simple RNN) | Clip Shuffle (Leaked) | ~64.20% (Overfit) | ~81.10% | 12.4 ms |
| Dual-Pooling BiLSTM (`lrcn-fpv16-bs64`) | Group-Aware Split | **83.77%** | **94.85%** | 18.2 ms |
| R(2+1)D-18 (`3dcnn-fpv16-bs64`) | Group-Aware Split | **91.13%** | **98.02%** | 24.6 ms |
| **Spatio-Temporal Ensemble (Soft-Voting)** | Group-Aware Split | **91.32%** | **98.90%** | 42.8 ms |

### Weights & Biases (W&B) Dashboard

Real-time convergence curves comparing the 3D-CNN (`3dcnn-fpv16-bs64`) and LRCN (`lrcn-fpv16-bs64`) runs over training and validation steps:

![W&B Training and Validation Metrics](assets/wandb_dashboard.png)

*Figure 1: W&B run comparisons for validation loss, validation accuracy, training loss, and training accuracy across 15 checkpoint steps.*

### Ensemble Evaluation Summary

```text
============================================================
Final Test Accuracy (Soft-Voting Ensemble): 91.3208%
============================================================
wandb: Run summary:
wandb: test/ensemble_accuracy 91.32075
```

---

## Dataset Preparation

1. Download the UCF50 dataset:
   ```bash
   wget https://www.crcv.ucf.edu/data/UCF50.rar --no-check-certificate
   unrar x UCF50.rar data/ucf50/
   ```

2. Verify directory layout:
   ```text
   data/ucf50/
   ├── BaseballPitch/
   │   ├── v_BaseballPitch_g01_c01.avi
   │   └── ...
   ├── Basketball/
   └── ...
   ```

---

## Environment Setup

Ensure you have Python 3.10+ and CUDA 11.8+ or 12.x installed.

```bash
# Clone the repository
git clone https://github.com/your-username/ucf50-action-recognition.git
cd ucf50-action-recognition

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Preprocessing and Frame Extraction

Extract decord-indexed metadata or pre-render resized frame sequences for accelerated IO:

```bash
python scripts/extract_frames.py \
    --raw_dir data/ucf50 \
    --output_dir data/ucf50_processed \
    --num_frames 16 \
    --image_size 224
```

---

## Training the Model

Train models with group-aware splits and mixed precision:

```bash
# Train Dual-Pooling BiLSTM LRCN
python train.py \
    --model lrcn \
    --backbone resnet50 \
    --run_name lrcn-fpv16-bs64 \
    --data_dir data/ucf50_processed \
    --epochs 35 \
    --batch_size 16 \
    --lr 1e-4 \
    --use_wandb \
    --mixed_precision

# Train R(2+1)D-18
python train.py \
    --model r2plus1d_18 \
    --run_name 3dcnn-fpv16-bs64 \
    --data_dir data/ucf50_processed \
    --epochs 45 \
    --batch_size 8 \
    --lr 5e-4 \
    --weight_decay 1e-4 \
    --use_wandb \
    --mixed_precision
```

---

## Testing and Evaluation

Evaluate checkpoints using 10-clip consensus and run soft-voting ensembling:

```bash
# Evaluate Single Model
python evaluate.py \
    --model r2plus1d_18 \
    --checkpoint checkpoints/r2plus1d_best.pth \
    --data_dir data/ucf50_processed \
    --num_clips 10 \
    --split test

# Evaluate Soft-Voting Ensemble (LRCN + 3D CNN)
python ensemble_eval.py \
    --checkpoints checkpoints/lrcn_best.pth checkpoints/r2plus1d_best.pth \
    --data_dir data/ucf50_processed \
    --num_clips 10 \
    --use_wandb
```

---

## Project Structure

```text
.
├── assets/
│   └── wandb_dashboard.png   # W&B dashboard metric plot
├── configs/                  # Experiment YAML configs
├── data/                     # Data manifests, raw, and cached clips
├── models/
│   ├── lrcn.py               # Dual-Pooling BiLSTM + 2D CNN architecture
│   └── r2plus1d.py           # Factorized 3D CNN architectures
├── scripts/
│   ├── extract_frames.py     # Frame extraction and caching pipeline
│   └── verify_splits.py      # Assert zero group overlap across partitions
├── utils/
│   ├── dataset.py            # GroupShuffleSplit dataset loaders
│   ├── metrics.py            # Accuracy, mAP, and Top-K metrics
│   └── transforms.py         # Video augmentations (spatial/temporal)
├── train.py                  # Training pipeline with AMP and grad clipping
├── evaluate.py               # Consensus evaluation and inference script
├── ensemble_eval.py          # Soft-voting multi-model evaluation script
├── requirements.txt          # Python runtime requirements
└── README.md                 # Project documentation
```

---

## Customization and Hyperparameters

Common CLI arguments configurable in `train.py` or `.yaml` configuration files:
- `--frames_per_clip`: Number of frames per temporal window (default: `16`).
- `--accum_steps`: Gradient accumulation steps for large effective batch sizes.
- `--dropout`: Recurrent/dense dropout probability (default: `0.5`).
- `--group_split_ratio`: Ratio for `[train, val, test]` partitions (default: `0.7, 0.15, 0.15`).

---

## Code Quality & Testing

Run unit tests verifying group split guarantees and network shape invariants:

```bash
pytest tests/
```
