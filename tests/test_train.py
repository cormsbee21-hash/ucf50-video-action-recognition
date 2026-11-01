"""
Unit tests for train.py helper functions.
"""
import torch
import torch.nn as nn
from train import batch_correct_preds, get_learning_rate, get_batch_loss


def test_batch_correct_preds():
    # 4 samples, 5 classes
    logits = torch.tensor([
        [2.5, 0.1, -1.0, 0.4, 0.2],   # class 0
        [0.0, 4.2, 1.1, 0.3, 0.0],    # class 1
        [-1.0, 0.5, 3.1, 0.2, 0.0],   # class 2
        [1.0, 0.5, 0.2, 0.0, 5.0]     # class 4
    ])
    
    # 3 correct, 1 incorrect
    targets = torch.tensor([0, 1, 2, 3])
    correct = batch_correct_preds(logits, targets)
    assert correct == 3


def test_get_learning_rate():
    dummy_param = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([dummy_param], lr=0.005)
    lr = get_learning_rate(optimizer)
    assert lr == 0.005


def test_get_batch_loss_backward():
    # Verify backpropagation updates parameters
    model = nn.Linear(10, 5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    inputs = torch.randn(4, 10)
    targets = torch.tensor([0, 1, 2, 3])

    initial_weight = model.weight.clone()
    output = model(inputs)
    loss, correct = get_batch_loss(criterion, output, targets, optimizer)

    assert isinstance(loss, float)
    assert 0 <= correct <= 4
    # Weights should update after optimizer.step()
    assert not torch.equal(model.weight, initial_weight)