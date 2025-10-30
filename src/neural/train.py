import numpy as np
from mnist_loader import combine_datasets, get_batch
from model import CNN

def train_cnn_with_operators(epochs=5, batch_size=32, learning_rate=0.01):
    """Train CNN on digits + operators (14 classes total)"""
    
    # Step 1: Load combined dataset
    print("Loading MNIST + Operators...")
    x_train, y_train, x_test, y_test = combine_datasets(
        mnist_path=r"C:\Users\konla\Desktop\PhD\mnist_data.npz",
        operator_path=r"C:\Users\konla\Desktop\PhD\Projects\cos521\neurosymbolic_mvp\src\neural\operators.npz"
    )
    
    # Step 2: Create CNN (now with 14 output classes)
    print("Initializing CNN for 14 classes...")
    model = CNN()
    
    # Rest of training loop stays the same...
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        batches_per_epoch = len(x_train) // batch_size
        
        for batch_idx in range(batches_per_epoch):
            x_batch, y_batch = get_batch(x_train, y_train, batch_size)
            logits = model.forward(x_batch)
            loss = model.compute_loss(logits, y_batch)
            epoch_loss += loss
            num_batches += 1
            model.backward()
            model.update_weights(learning_rate)
            
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{batches_per_epoch}, Loss: {loss:.4f}")
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")
        
        # Evaluate on test set
        print("  Evaluating on test set...")
        test_accuracy = evaluate(model, x_test, y_test, batch_size)
        print(f"  Test Accuracy: {test_accuracy:.2f}%\n")
    
    print("Training completed!")
    return model

def evaluate(model, x_test, y_test, batch_size=32):
    """
    Evaluate model accuracy on test set.
    
    Args:
        model: trained CNN model
        x_test: test images
        y_test: test labels
        batch_size: batch size for evaluation
    
    Returns:
        accuracy: percentage of correct predictions
    """
    correct = 0
    total = 0
    
    # Evaluate in batches to avoid memory issues
    num_batches = len(x_test) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        x_batch = x_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]
        
        # Get predictions
        predictions = model.predict(x_batch)
        
        # Count correct predictions
        correct += np.sum(predictions == y_batch)
        total += len(y_batch)
    
    accuracy = (correct / total) * 100
    return accuracy

if __name__ == "__main__":
    model = train_cnn_with_operators(epochs=3, batch_size=32, learning_rate=0.01)
    
    # Save the trained model
    model.save_weights('trained_cnn_model.pkl')
    print("\n✓ Model saved successfully!")