"""
Unit tests for models.py verifying batch_first handling and output shapes.
"""
import torch
from models import LRCN


def test_baseline_lrcn_output_shape():
    batch_size = 2
    time_steps = 16
    channels = 3
    height, width = 224, 224
    num_classes = 50

    dummy_input = torch.randn(batch_size, time_steps, channels, height, width)

    model = LRCN(
        hidden_size=128,
        n_layers=1,
        dropout_rate=0.2,
        n_classes=num_classes,
        pretrained=False,
        cnn_model='resnet18'
    )
    model.eval()

    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (batch_size, num_classes), f"Expected {(batch_size, num_classes)}, got {out.shape}"