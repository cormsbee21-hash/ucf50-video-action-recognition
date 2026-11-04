"""
Module: models.py

Defines video classification architectures for UCF50:
  - 3D CNN: Factorized Spatio-Temporal ConvNet (R(2+1)D-18) pretrained on Kinetics-400.
  - LRCN: 2D ResNet-34 backbone with Bidirectional LSTM and dual temporal pooling.
"""

import torch
from torch import nn
from torchvision import models
import torchvision.models.video as video_models


class Identity(nn.Module):
    """Placeholder identity operator replacing base CNN classification heads."""

    def forward(self, x):
        """Pass input through unchanged."""
        return x


class VideoResNet3D(nn.Module):
    """
    Factorized Spatio-temporal ConvNet (R(2+1)D-18) pretrained on Kinetics-400.
    Splits 3D convolution into spatial (1x3x3) and temporal (3x1x1) components
    with intermediate ReLU non-linearities for higher representational capacity.
    """

    def __init__(self, n_classes=50, pretrained=True):
        super().__init__()
        weights = video_models.R2Plus1D_18_Weights.DEFAULT if pretrained else None
        self.backbone = video_models.r2plus1d_18(weights=weights)

        # Freeze stem and low-level blocks (conv1, layer1, layer2)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze layer3 (mid-level motion) and layer4 (high-level semantics)
        for param in self.backbone.layer3.parameters():
            param.requires_grad = True
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_feats, n_classes)
        )

    def forward(self, x):
        """
        Input x: (batch_size, channels, time_steps, height, width)
        """
        return self.backbone(x)


class LRCN(nn.Module):
    """
    Enhanced LRCN integrating a 2D ResNet backbone with a Bidirectional LSTM
    and dual temporal pooling (Average + Max).
    """

    def __init__(
        self,
        hidden_size=256,
        n_layers=2,
        dropout_rate=0.3,
        n_classes=50,
        **kwargs
    ):
        super().__init__()
        pretrained = kwargs.get("pretrained", True)
        cnn_model = kwargs.get("cnn_model", "resnet34")

        if cnn_model == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_cnn = models.resnet18(weights=weights)
        elif cnn_model == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_cnn = models.resnet50(weights=weights)
        else:
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            base_cnn = models.resnet34(weights=weights)

        num_features = base_cnn.fc.in_features
        base_cnn.fc = Identity()
        self.base_model = base_cnn

        # Freeze early layers (conv1, bn1, layer1, layer2)
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Fine-tune high-level visual representations
        for param in self.base_model.layer3.parameters():
            param.requires_grad = True
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        self.rnn = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout_rate)
        # Bidirectional (* 2) + Dual Pooling (Avg + Max = * 2) -> hidden_size * 4
        self.fc = nn.Linear(hidden_size * 4, n_classes)

    def forward(self, x):
        """
        Extract spatial representations per frame and model temporal dynamics.
        Input x: (batch_size, time_steps, channels, height, width)
        """
        bs, ts, c, h, w = x.shape

        c_in = x.reshape(bs * ts, c, h, w)
        cnn_features = self.base_model(c_in)
        r_in = cnn_features.view(bs, ts, -1)

        r_out, _ = self.rnn(r_in)
        avg_pool = torch.mean(r_out, dim=1)
        max_pool = torch.amax(r_out, dim=1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        out = self.dropout(pooled)
        return self.fc(out)
