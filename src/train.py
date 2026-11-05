"""Train a reproducible PointNet-style 3D shape classifier."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from dataset import PointCloudDataset
from model import PointNetClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/synthetic_shapes.npz')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='runs/pointnet')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = PointCloudDataset(args.data)
    if len(dataset) < 5:
        raise ValueError('At least 5 samples required')
    indices = np.random.default_rng(args.seed).permutation(len(dataset))
    n_val = max(1, int(len(indices) * 0.2))
    train_ds = Subset(dataset, indices[n_val:].tolist())
    val_ds = Subset(dataset, indices[:n_val].tolist())
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNetClassifier(num_classes=int(dataset.labels.max()) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history, best_acc = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for points, labels in train_loader:
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(points), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        model.eval()
        correct = count = 0
        with torch.no_grad():
            for points, labels in val_loader:
                predictions = model(points.to(device)).argmax(dim=1)
                correct += (predictions == labels.to(device)).sum().item()
                count += len(labels)
        accuracy = correct / count
        record = {'epoch': epoch, 'train_loss': total_loss / len(train_ds), 'val_accuracy': accuracy}
        history.append(record)
        print(json.dumps(record))
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({'model_state_dict': model.state_dict(), 'num_classes': model.head[-1].out_features}, output / 'best_model.pt')
    (output / 'metrics.json').write_text(json.dumps({'best_val_accuracy': best_acc, 'history': history, 'seed': args.seed, 'device': str(device)}, indent=2))
    print(f'Saved checkpoint and metrics in {output}')


if __name__ == '__main__':
    main()
