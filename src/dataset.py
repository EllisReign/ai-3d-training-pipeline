"""Point cloud dataset loader and deterministic synthetic example generator."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    def __init__(self, path: str, augment: bool = False):
        with np.load(path, allow_pickle=False) as data:
            self.points = data['points'].astype(np.float32)
            self.labels = data['labels'].astype(np.int64)
        if self.points.ndim != 3 or self.points.shape[-1] != 3:
            raise ValueError("points must have shape [samples, points, 3]")
        if self.labels.ndim != 1 or len(self.labels) != len(self.points):
            raise ValueError("labels must have shape [samples]")
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        points = self.points[index].copy()
        if self.augment:
            angle = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            points = points @ rotation.T
            points += np.random.normal(0, 0.01, points.shape).astype(np.float32)
        return torch.from_numpy(points), torch.tensor(self.labels[index], dtype=torch.long)


def generate_synthetic(output: str, samples_per_class: int = 120, n_points: int = 256, seed: int = 42):
    """Generate toy spheres, cubes, and cylinders; NOT biomedical measurements."""
    rng = np.random.default_rng(seed)
    point_clouds, labels = [], []
    for label in range(3):
        for _ in range(samples_per_class):
            if label == 0:  # surface of sphere
                p = rng.normal(size=(n_points, 3))
                p /= np.linalg.norm(p, axis=1, keepdims=True).clip(min=1e-8)
            elif label == 1:  # cube volume
                p = rng.uniform(-1, 1, (n_points, 3))
            else:  # cylindrical volume
                theta = rng.uniform(0, 2 * np.pi, n_points)
                r = np.sqrt(rng.uniform(0, 1, n_points))
                p = np.stack([r * np.cos(theta), r * np.sin(theta), rng.uniform(-1, 1, n_points)], axis=-1)
            p = p + rng.normal(0, 0.015, p.shape)
            point_clouds.append(p.astype(np.float32))
            labels.append(label)
    permutation = rng.permutation(len(labels))
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, points=np.stack(point_clouds)[permutation], labels=np.array(labels)[permutation])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/synthetic_shapes.npz')
    parser.add_argument('--samples-per-class', type=int, default=120)
    parser.add_argument('--n-points', type=int, default=256)
    args = parser.parse_args()
    generate_synthetic(args.output, args.samples_per_class, args.n_points)
    print(f'Generated synthetic dataset: {args.output}')
