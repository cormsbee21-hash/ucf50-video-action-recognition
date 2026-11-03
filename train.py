"""
Module: train.py

This module provides functions to train and evaluate a video classification model.
It includes the main training loop that tracks loss and accuracy over epochs,
updates the best performing model based on validation accuracy, and provides helper
functions to compute the learning rate, batch loss, and epoch loss.
"""

import os
import copy
from tqdm import tqdm
import torch
from torch.amp import autocast, GradScaler
import wandb


def _record_metrics(metrics, best_state, save_dir, epoch):
    """Log metrics to console and wandb, saving checkpoint if accuracy improved."""
    train_loss, train_acc, val_loss, val_acc, current_lr = metrics
    best_acc, best_wts = best_state

    if val_acc > best_acc:
        best_acc = val_acc
        best_wts = copy.deepcopy(best_wts)
        torch.save(best_wts, os.path.join(save_dir, 'best_model_wts.pt'))
        print(f"Best model weights are updated at epoch {epoch + 1}!")

    if wandb.run is not None:
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "learning_rate": current_lr
        })

    print(
        f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, "
        f"accuracy: {100 * val_acc:.2f}"
    )
    print("-" * 60)
    print()
    return best_acc, best_wts


def _run_epoch(model, dataloaders, opt_setup, device):
    """Run one epoch of training and validation."""
    train_step = {'optimizer': opt_setup['optimizer'], 'scaler': GradScaler('cuda')}
    criterion = opt_setup['criterion']

    model.train()
    tr_loss, tr_acc = get_epoch_loss(
        model, criterion, dataloaders['train'], device, train_step=train_step
    )

    model.eval()
    with torch.no_grad():
        val_loss, val_acc = get_epoch_loss(
            model, criterion, dataloaders['val'], device
        )

    return tr_loss, tr_acc, val_loss, val_acc


def train(dataloaders, model, opt_setup, train_cfg):
    """
    Train and validate the model over a given number of epochs.

    Args:
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        model (torch.nn.Module): The video classification model.
        opt_setup (dict): Dictionary with 'criterion', 'optimizer', 'scheduler'.
        train_cfg (dict): Dictionary with 'device', 'save_dir', and 'n_epochs'.
    """
    hist = {'tr_loss': [], 'tr_acc': [], 'val_loss': [], 'val_acc': []}
    best_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(train_cfg.get('n_epochs', 30)):
        current_lr = get_learning_rate(opt_setup['optimizer'])
        print(
            f"Epoch {epoch + 1}/{train_cfg.get('n_epochs', 30)}; "
            f"Current learning rate {current_lr}"
        )

        epoch_results = _run_epoch(model, dataloaders, opt_setup, train_cfg['device'])
        for key, val in zip(['tr_loss', 'tr_acc', 'val_loss', 'val_acc'], epoch_results):
            hist[key].append(val)

        metrics = (*epoch_results, current_lr)
        best_val_acc, best_wts = _record_metrics(
            metrics, (best_val_acc, model.state_dict()), train_cfg['save_dir'], epoch
        )
        opt_setup['scheduler'].step(epoch_results[2])

    model.load_state_dict(best_wts)
    return model, hist


def get_learning_rate(optimizer):
    """Retrieve the current learning rate from the optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def batch_correct_preds(output, target):
    """Compute the number of correct predictions for a mini-batch."""
    pred = output.argmax(dim=1, keepdim=True)
    return pred.eq(target.view_as(pred)).sum().item()


def get_batch_loss(criterion, output, target, train_step=None):
    """Compute loss and correct predictions, executing backprop if training."""
    loss = criterion(output, target)
    n_correct = batch_correct_preds(output, target)

    if train_step is not None:
        if isinstance(train_step, dict):
            opt = train_step['optimizer']
            scaler = train_step.get('scaler')
        else:
            opt = train_step
            scaler = None

        opt.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

    return loss.item(), n_correct


def get_epoch_loss(model, criterion, dataloader, device, train_step=None):
    """Compute the average loss and overall accuracy for an epoch."""
    running_loss, running_correct = 0.0, 0
    total_samples = len(dataloader.dataset)

    for x_batch, y_batch in tqdm(dataloader):
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        with autocast('cuda', enabled=device.type == 'cuda'):
            output = model(x_batch)
            b_loss, n_correct = get_batch_loss(
                criterion, output, y_batch, train_step=train_step
            )

        running_loss += b_loss * x_batch.size(0)
        running_correct += n_correct

    return running_loss / float(total_samples), running_correct / float(total_samples)
