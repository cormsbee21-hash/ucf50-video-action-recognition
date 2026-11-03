"""
Module: test.py

This module provides functions for evaluating trained video classification models.
It includes functions to run model inference on a test dataset, compute accuracy,
generate classification reports, and calculate confusion matrices.
"""

from tqdm import tqdm
import torch
from sklearn.metrics import classification_report, multilabel_confusion_matrix


def test(model, test_dataloader, device):
    """
    Evaluate the model on the test dataset.

    Args:
        model (torch.nn.Module): Trained model to evaluate.
        test_dataloader (DataLoader): DataLoader for test data.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (targets, outputs, accuracy)
            - targets (list): Ground truth labels.
            - outputs (list): Predicted class labels.
            - accuracy (float): Test set accuracy.
    """
    running_corrects = 0
    targets = []
    outputs = []

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(x_batch)
            preds = output.argmax(dim=1)

            running_corrects += preds.eq(y_batch).sum().item()
            targets.extend(y_batch.tolist())
            outputs.extend(preds.tolist())

    accuracy = running_corrects / float(len(test_dataloader.dataset))
    return targets, outputs, accuracy


def get_test_report(targets, outputs, all_cats):
    """
    Generate a text classification report showing precision, recall, and F1-score.

    Args:
        targets (list): Ground truth class labels.
        outputs (list): Predicted class labels.
        all_cats (list): List of all class names.

    Returns:
        dict: A dictionary containing precision, recall, and f1-scores for each class.
    """
    report = classification_report(
        targets,
        outputs,
        target_names=all_cats,
        output_dict=True
    )
    return report


def get_confusion_matrix(targets, outputs, labels_dict, all_cats):
    """
    Calculate multilabel confusion matrices for each class.

    Args:
        targets (list): Ground truth class labels.
        outputs (list): Predicted class labels.
        labels_dict (dict): Dictionary mapping class names to integer labels.
        all_cats (list): List of all class names.

    Returns:
        dict: A dictionary where keys are class names and values are the
        corresponding confusion matrices.
    """
    inv_labels_dict = {label: cat for cat, label in labels_dict.items()}
    target_cats = [inv_labels_dict[target] for target in targets]
    output_cats = [inv_labels_dict[output] for output in outputs]
    confusion_mat = multilabel_confusion_matrix(
        target_cats, output_cats, labels=all_cats
    )
    return dict(zip(all_cats, confusion_mat))
