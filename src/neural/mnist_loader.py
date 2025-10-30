import numpy as np

def load_mnist_npz(path: str):
    """
    Loads MNIST data from .npz file.
    Returns:
        x_train: (N_train, 28, 28), y_train: (N_train,)
        x_test: (N_test, 28, 28), y_test: (N_test,)
    """
    data = np.load(path)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    # Normalize to [0,1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    # Add channel dimension for CNN: (N, 1, 28, 28)
    x_train = x_train[:, None, :, :]
    x_test = x_test[:, None, :, :]
    return x_train, y_train, x_test, y_test

def load_operators(path: str):
    """Load operator dataset"""
    data = np.load(path)
    images = data['images']
    labels = data['labels']
    return images, labels

def combine_datasets(mnist_path: str, operator_path: str, test_split=0.2):
    """
    Combine MNIST digits (0-9) with operators (+, -, ×, ÷).
    Returns combined train/test sets with 14 classes (0-9, 10-13).
    """
    # Load MNIST
    x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist = load_mnist_npz(mnist_path)
    
    # Load operators
    x_operators, y_operators = load_operators(operator_path)
    
    # Split operators into train/test
    n_test = int(len(x_operators) * test_split)
    indices = np.random.permutation(len(x_operators))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    x_train_ops = x_operators[train_idx]
    y_train_ops = y_operators[train_idx]
    x_test_ops = x_operators[test_idx]
    y_test_ops = y_operators[test_idx]
    
    # Combine datasets
    x_train = np.concatenate([x_train_mnist, x_train_ops], axis=0)
    y_train = np.concatenate([y_train_mnist, y_train_ops], axis=0)
    x_test = np.concatenate([x_test_mnist, x_test_ops], axis=0)
    y_test = np.concatenate([y_test_mnist, y_test_ops], axis=0)
    
    # Shuffle
    train_indices = np.random.permutation(len(x_train))
    test_indices = np.random.permutation(len(x_test))
    
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    x_test = x_test[test_indices]
    y_test = y_test[test_indices]
    
    print(f"Combined dataset: {len(x_train)} train, {len(x_test)} test samples")
    print(f"Classes: 0-9 (digits), 10 (+), 11 (-), 12 (×), 13 (÷)")
    
    return x_train, y_train, x_test, y_test

def get_batch(x, y, batch_size=32):
    """Get a random batch from the dataset"""
    idx = np.random.choice(len(x), batch_size, replace=False)
    return x[idx], y[idx]