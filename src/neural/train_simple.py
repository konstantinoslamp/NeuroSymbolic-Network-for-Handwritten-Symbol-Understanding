"""
Simple CNN training script that works both locally and in Google Colab
Uses MNIST only (no operators) for basic validation
"""

import numpy as np
import os
import sys

# Make sure we can import from parent directories
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from model import CNN
except ImportError:
    print("ERROR: Cannot import CNN from model.py")
    print("Make sure you're in the neurosymbolic_mvp directory")
    sys.exit(1)


def load_mnist_simple(mnist_path=None):
    """
    Load MNIST dataset from npz file.
    If file doesn't exist locally, try to download it.
    """
    if mnist_path is None:
        # Try multiple possible paths
        possible_paths = [
            'src/neural/mnist.npz',
            'mnist.npz',
            '/content/neurosymbolic_mvp/src/neural/mnist.npz'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                mnist_path = path
                break
        
        if mnist_path is None:
            print("ERROR: mnist.npz not found in expected locations")
            print("Attempted paths:")
            for p in possible_paths:
                print(f"  - {p}")
            raise FileNotFoundError("MNIST data not found")
    
    print(f"Loading MNIST from: {mnist_path}")
    
    # Load
    data = np.load(mnist_path)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    
    # Normalize to [0,1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # Add channel dimension: (N, 1, 28, 28)
    x_train = x_train[:, None, :, :]
    x_test = x_test[:, None, :, :]
    
    print(f"✓ MNIST loaded: {len(x_train)} train, {len(x_test)} test")
    
    return x_train, y_train, x_test, y_test


def get_batch(x, y, batch_size=32):
    """Get a random batch"""
    idx = np.random.choice(len(x), batch_size, replace=False)
    return x[idx], y[idx]


def train_cnn_simple(epochs=2, batch_size=32, learning_rate=0.01):
    """
    Train basic CNN on MNIST (10 classes).
    
    Args:
        epochs: number of training epochs
        batch_size: batch size
        learning_rate: learning rate for updates
    """
    
    print("\n" + "="*70)
    print("SIMPLE CNN TRAINING")
    print("="*70)
    
    # Load data
    print("\n[1/3] Loading dataset...")
    x_train, y_train, x_test, y_test = load_mnist_simple()
    
    # Create model
    print("\n[2/3] Initializing CNN...")
    model = CNN()
    print(f"✓ CNN initialized for 10 classes (MNIST digits)")
    
    # Train
    print("\n[3/3] Training...")
    print("-" * 70)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        batches_per_epoch = len(x_train) // batch_size
        
        for batch_idx in range(batches_per_epoch):
            # Get batch
            x_batch, y_batch = get_batch(x_train, y_train, batch_size)
            
            # Forward pass
            logits = model.forward(x_batch)
            loss = model.compute_loss(logits, y_batch)
            epoch_loss += loss
            num_batches += 1
            
            # Backward pass
            model.backward()
            model.update_weights(learning_rate)
            
            # Progress
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{batches_per_epoch}, Loss: {loss:.4f}")
        
        # Average loss for epoch
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")
        
        # Evaluate on test set
        print("  Evaluating on test set...")
        correct = 0
        total = 0
        
        num_test_batches = len(x_test) // batch_size
        for i in range(num_test_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            x_batch = x_test[start_idx:end_idx]
            y_batch = y_test[start_idx:end_idx]
            
            predictions = model.predict(x_batch)
            correct += np.sum(predictions == y_batch)
            total += len(y_batch)
        
        test_accuracy = (correct / total) * 100
        print(f"  Test Accuracy: {test_accuracy:.2f}%\n")
    
    print("="*70)
    print("Training completed!")
    print("="*70)
    
    # Save model
    save_path = 'trained_cnn_model.pkl'
    print(f"\nSaving model to: {save_path}")
    model.save_weights(save_path)
    print(f"✓ Model saved successfully!")
    
    return model


if __name__ == "__main__":
    try:
        model = train_cnn_simple(epochs=2, batch_size=32, learning_rate=0.01)
        print("\n✅ SUCCESS: CNN training complete!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
