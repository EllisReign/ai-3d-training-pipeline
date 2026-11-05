import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dataset import generate_synthetic, PointCloudDataset
from model import PointNetClassifier


def test_dataset(tmp_path):
    path = tmp_path / 'shapes.npz'
    generate_synthetic(str(path), samples_per_class=3, n_points=32)
    dataset = PointCloudDataset(str(path))
    assert len(dataset) == 9
    points, label = dataset[0]
    assert points.shape == (32, 3)
    assert label.dtype == torch.long
    assert set(dataset.labels.tolist()) == {0, 1, 2}


def test_model_forward_and_backward():
    model = PointNetClassifier(3)
    x = torch.randn(4, 32, 3)
    outputs = model(x)
    assert outputs.shape == (4, 3)
    outputs.sum().backward()
    assert any(p.grad is not None for p in model.parameters())
