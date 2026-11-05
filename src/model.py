"""A lightweight PointNet-style classifier for 3D point clouds."""
import torch
from torch import nn


class PointNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("Expected point clouds with shape [batch, points, 3]")
        features = self.encoder(points.transpose(1, 2))
        return self.head(features.max(dim=2).values)
