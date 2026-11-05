"""Predict 3D shape class from an NPZ point cloud file."""
import argparse
import numpy as np
import torch
from model import PointNetClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='runs/pointnet/best_model.pt')
    parser.add_argument('--data', default='data/synthetic_shapes.npz')
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = PointNetClassifier(checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with np.load(args.data, allow_pickle=False) as data:
        points = data['points'][args.index].astype(np.float32)
    with torch.no_grad():
        probs = model(torch.from_numpy(points).unsqueeze(0)).softmax(dim=-1)[0]
    print({'predicted_class': int(probs.argmax()), 'confidence': float(probs.max()), 'class_names': ['sphere', 'cube', 'cylinder']})


if __name__ == '__main__':
    main()
