"""
Module: models.py

Defines video classification architectures for UCF50.
Supports:
  - 3D CNN: Inflated 3D Spatio-Temporal ResNet (R3D-18), per Module 7 Lecture Slides 70-73.
  - LRCN: 2D ResNet-34 backbone with Bidirectional LSTM and dual temporal pooling.
"""
import torch
from torch import nn
from torchvision import models
import torchvision.models.video as video_models


class Identity(nn.Module):
    """Placeholder identity operator replacing the base CNN classification head."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Pass input through unchanged."""
        return x


class VideoResNet3D(nn.Module):
    """
    Spatio-temporal 3D CNN (R3D-18) pretrained on Kinetics-400.
    Directly extracts spatio-temporal features across (C, T, H, W) volumes.
    Lecture Reference: Module 7 Slides 70-73 (Inflating 2D Networks to 3D).
    """
    def __init__(self, n_classes=50, pretrained=True):
        super().__init__()
        weights = video_models.R3D_18_Weights.DEFAULT if pretrained else None
        self.backbone = video_models.r3d_18(weights=weights)

        # Freeze early spatio-temporal layers (conv1 through layer3)
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Adapt high-level layer4 spatio-temporal residual block
        for param in self.backbone.layer3.parameters():
            param.requires_grad = True
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_feats, n_classes)
        )

    def forward(self, x):
        """
        Input x: (batch_size, channels, time_steps, height, width) from train_val_dloaders
        """
        return self.backbone(x)


class LRCN(nn.Module):
    """
    Enhanced LRCN integrating a 2D ResNet backbone with a Bidirectional LSTM
    and dual temporal pooling (Average + Max).
    """
    def __init__(self, hidden_size, n_layers, dropout_rate, n_classes, **kwargs):
        super().__init__()
        pretrained = kwargs.get('pretrained', True)
        cnn_model = kwargs.get('cnn_model', 'resnet34')

        if cnn_model == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_cnn = models.resnet18(weights=weights)
        elif cnn_model == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_cnn = models.resnet50(weights=weights)
        else:
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            base_cnn = models.resnet34(weights=weights)

        num_features = base_cnn.fc.in_features
        base_cnn.fc = Identity()
        self.base_model = base_cnn

        # Freeze early layers, keep layer3 and layer4 trainable
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.base_model.layer3.parameters():
            param.requires_grad = True
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        self.rnn = nn.LSTM(
            num_features,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 4, n_classes)

    def forward(self, x):
        """
        Extract spatial representations per frame and model sequence dynamics with BiLSTM.
        """
        bs, ts, c, h, w = x.shape
        self.base_model.eval()

        c_in = x.reshape(bs * ts, c, h, w)
        cnn_features = self.base_model(c_in)
        r_in = cnn_features.view(bs, ts, -1)

        r_out, _ = self.rnn(r_in)
        avg_pool = torch.mean(r_out, dim=1)
        max_pool = torch.amax(r_out, dim=1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        out = self.dropout(pooled)
        return self.fc(out)
