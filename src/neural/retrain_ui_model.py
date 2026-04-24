"""
Retrain the UI recognition model using PyTorch + torchvision MNIST.

Fixes the normalization mismatch in the original trained_cnn_model.pkl:
  - Original: MNIST in [0,1], operators in [-0.49, 1.46] (inconsistent)
  - New: both in [0,1], same preprocessing as the UI canvas

Run from project root:
    python src/neural/retrain_ui_model.py

Saves: src/neural/ui_cnn_model.pt  (PyTorch state dict)
       src/neural/ui_cnn_model.pkl  (drop-in replacement for trained_cnn_model.pkl)
Takes: ~2-3 minutes on CPU, ~30s on GPU.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class UICnn(nn.Module):
    """Simple CNN for 14-class symbol recognition (digits 0-9 + + - x ÷)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 28x28 -> 28x28
        self.pool  = nn.MaxPool2d(2, 2)               # -> 14x14 then 7x7
        self.fc1   = nn.Linear(32 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 14)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mnist():
    """Download MNIST via torchvision and return numpy arrays in [0,1]."""
    try:
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.MNIST(
            root=os.path.join(SCRIPT_DIR, '_mnist_cache'),
            train=True, download=True, transform=transform)
        test_set  = torchvision.datasets.MNIST(
            root=os.path.join(SCRIPT_DIR, '_mnist_cache'),
            train=False, download=True, transform=transform)

        x_train = train_set.data.numpy().astype(np.float32) / 255.0   # (60000, 28, 28)
        y_train = train_set.targets.numpy()
        x_test  = test_set.data.numpy().astype(np.float32)  / 255.0   # (10000, 28, 28)
        y_test  = test_set.targets.numpy()
        print(f'MNIST loaded: {len(x_train)} train, {len(x_test)} test')
        return x_train[:, None], y_train, x_test[:, None], y_test      # add channel dim
    except Exception as e:
        print(f'torchvision MNIST failed: {e}')
        return None


def load_operators_normalized():
    """
    Load operators.npz and normalize to [0,1].
    The file was saved with (x - 0.25) / 0.514 normalization applied.
    We reverse it and clip to [0,1].
    """
    path = os.path.join(SCRIPT_DIR, 'operators.npz')
    data = np.load(path)
    imgs = data['images'].astype(np.float32)   # (N, 1, 28, 28) in [-0.49, 1.46]
    labels = data['labels']

    # Reverse the per-dataset normalization and clip to [0,1]
    imgs = imgs * 0.514 + 0.250
    imgs = np.clip(imgs, 0.0, 1.0)
    print(f'Operators loaded: {len(imgs)} samples, range [{imgs.min():.2f}, {imgs.max():.2f}]')
    return imgs, labels


def augment_operators(imgs, labels, factor=5):
    """
    Augment operator images with random shifts and noise so the model
    sees variety similar to hand-drawn input.
    """
    aug_imgs, aug_labels = [], []
    for img, lbl in zip(imgs, labels):
        aug_imgs.append(img)
        aug_labels.append(lbl)
        for _ in range(factor - 1):
            # Random pixel shift (±2px)
            shift_x = np.random.randint(-2, 3)
            shift_y = np.random.randint(-2, 3)
            shifted = np.roll(np.roll(img, shift_x, axis=2), shift_y, axis=1)
            # Small noise
            noisy = shifted + np.random.normal(0, 0.05, shifted.shape).astype(np.float32)
            noisy = np.clip(noisy, 0.0, 1.0)
            aug_imgs.append(noisy)
            aug_labels.append(lbl)
    return np.array(aug_imgs), np.array(aug_labels)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(epochs=10, batch_size=64, lr=1e-3):
    # Load data
    mnist = load_mnist()
    if mnist is None:
        print('ERROR: Could not load MNIST. Install torchvision: pip install torchvision')
        sys.exit(1)
    x_tr_m, y_tr_m, x_te_m, y_te_m = mnist

    x_ops, y_ops = load_operators_normalized()
    x_ops_aug, y_ops_aug = augment_operators(x_ops, y_ops, factor=6)

    # Split operators into train/test
    n = len(x_ops_aug)
    perm = np.random.permutation(n)
    split = int(n * 0.8)
    x_tr_o, y_tr_o = x_ops_aug[perm[:split]], y_ops_aug[perm[:split]]
    x_te_o, y_te_o = x_ops_aug[perm[split:]], y_ops_aug[perm[split:]]

    # Combine
    x_train = np.concatenate([x_tr_m, x_tr_o], axis=0)
    y_train = np.concatenate([y_tr_m, y_tr_o], axis=0)
    x_test  = np.concatenate([x_te_m, x_te_o], axis=0)
    y_test  = np.concatenate([y_te_m, y_te_o], axis=0)
    print(f'Combined train: {len(x_train)}, test: {len(x_test)}')

    # Shuffle
    perm = np.random.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]

    # Datasets
    train_ds = TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train))
    test_ds  = TensorDataset(torch.FloatTensor(x_test),  torch.LongTensor(y_test))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=256, shuffle=False)

    # Model
    model = UICnn().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(f'\nTraining for {epochs} epochs...')
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(xb)
        scheduler.step()

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_correct += (model(xb).argmax(1) == yb).sum().item()
                val_total += len(xb)

        print(f'  Epoch {epoch}/{epochs}  '
              f'loss={total_loss/total:.4f}  '
              f'train_acc={correct/total:.3f}  '
              f'val_acc={val_correct/val_total:.3f}')

    # Save PyTorch model
    pt_path = os.path.join(SCRIPT_DIR, 'ui_cnn_model.pt')
    torch.save(model.state_dict(), pt_path)
    print(f'\nSaved PyTorch model -> {pt_path}')

    # Also save as pkl wrapper so the UI can load it without changes
    pkl_path = os.path.join(SCRIPT_DIR, 'ui_cnn_model.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({'type': 'pytorch', 'state_dict': model.state_dict()}, f)
    print(f'Saved pkl wrapper  -> {pkl_path}')

    return model


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    model = train(epochs=10)
    print('\nDone. Update ui_app.py model_path to use ui_cnn_model.pt')
    print('Or run: python src/ui_app.py  (it will auto-detect the new model)')
