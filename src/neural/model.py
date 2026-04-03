import numpy as np
from src.neural.cnn import Conv2D, MaxPool2D, Flatten, Dense, ReLU, SoftmaxCrossEntropy
import pickle 

class CNN:
    """
    A simple CNN for MNIST digit classification.
    Architecture:
      Conv(1→8, 3×3) → ReLU → MaxPool(2×2) → Flatten → Dense(8*13*13→128) → ReLU → Dense(128→10) → Softmax
    """
    
    def __init__(self):
        # Layers
        # Create a convolutional layer: 1 input channel (greyscale), 8 output filters, 3x3 kernel
        self.conv1 = Conv2D(in_channels=1, out_channels=8, kernel_size=3)
        # ReLU activation after convolution (introduces non-linearity)
        self.relu1 = ReLU()
        # Max pooling layer to reduce spatial dimensions
        self.pool1 = MaxPool2D()
        # Flattern layer to convert 2D feature maps to 1D feature vectors
        self.flatten = Flatten()
        # After conv (28→26) and pool (26→13): feature map size is 8 channels × 13 × 13 = 1352 features
        
        # First fully-connected layer: 1352 inputs → 128 hidden units
        self.fc1 = Dense(in_features=8*13*13, out_features=128)
        # ReLU activation after first fully-connected layer
        self.relu2 = ReLU()
        # Second fully-connected layer: 128 inputs → 14 outputs (one per digit class 0-9)
        self.fc2 = Dense(in_features=128, out_features=14)
        # Store layers with trainable parameters (weights/biases) for easy updates during trainin
        self.loss_fn = SoftmaxCrossEntropy()
        
        # Store all trainable layers
        self.trainable_layers = [self.conv1, self.fc1, self.fc2]

    def forward(self, x):
        """
        Forward pass: run input through all layers to get predictions.
        x: (N, 1, 28, 28) - batch of N grayscale 28×28 images
        returns: logits (N, 10) - raw scores for each of 10 digit classes
        """
        # Apply convolution: (N, 1, 28, 28) → (N, 8, 26, 26)
        x = self.conv1.forward(x)
        
        # Apply ReLU: element-wise max(0, x)
        x = self.relu1.forward(x)
        
        # Apply max pooling: (N, 8, 26, 26) → (N, 8, 13, 13)
        x = self.pool1.forward(x)
        
        # Flatten: (N, 8, 13, 13) → (N, 1352)
        x = self.flatten.forward(x)
        
        # First dense layer: (N, 1352) → (N, 128)
        x = self.fc1.forward(x)
        
        # Apply ReLU activation
        x = self.relu2.forward(x)
        
        # Second dense layer (output layer): (N, 128) → (N, 10)
        x = self.fc2.forward(x)
        
        # Return raw logits (before softmax - loss function will handle softmax)
        return x

    def compute_loss(self, logits, labels):
        """
        Compute the loss between predictions and true labels.
        logits: (N, 10) - raw scores from forward pass
        labels: (N,) - true digit labels (integers 0-9)
        returns: scalar loss value
        """
        return self.loss_fn.forward(logits, labels)

    def backward(self, grad=None):
        """
        Backward pass: compute gradients by backpropagating through all layers.
        Starts from loss and goes backwards through the network.
        """
        # Get gradient from loss function (w.r.t. logits)
        if grad is None:
            grad = self.loss_fn.backward()
        
        # Backprop through second dense layer
        grad = self.fc2.backward(grad)
        
        # Backprop through second ReLU
        grad = self.relu2.backward(grad)
        
        # Backprop through first dense layer
        grad = self.fc1.backward(grad)
        
        # Backprop through flatten (just reshapes gradient)
        grad = self.flatten.backward(grad)
        
        # Backprop through max pooling
        grad = self.pool1.backward(grad)
        
        # Backprop through first ReLU
        grad = self.relu1.backward(grad)
        
        # Backprop through convolution layer
        grad = self.conv1.backward(grad)

    def update_weights(self, learning_rate=0.01):
        """
        Update all trainable parameters using simple SGD (Stochastic Gradient Descent).
        learning_rate: how much to adjust weights (step size)
        """
        # Loop through each layer with trainable parameters
        for layer in self.trainable_layers:
            # Check if layer has computed gradients
            if hasattr(layer, 'grad_W'):
                # Update weights: W = W - learning_rate × gradient
                layer.W -= learning_rate * layer.grad_W
                # Update biases: b = b - learning_rate × gradient
                layer.b -= learning_rate * layer.grad_b

    def predict(self, x):
        """
        Make predictions on input data.
        x: (N, 1, 28, 28) - batch of images
        returns: (N,) - predicted digit class for each image (0-9)
        """
        # Run forward pass to get logits
        logits = self.forward(x)
        # Return class with highest score (argmax across 10 classes)
        return np.argmax(logits, axis=1)

    def save_weights(self, filepath: str):
        """Save model weights to file"""
        weights = {
            'conv1_W': self.conv1.W,
            'conv1_b': self.conv1.b,
            'fc1_W': self.fc1.W,
            'fc1_b': self.fc1.b,
            'fc2_W': self.fc2.W,
            'fc2_b': self.fc2.b
        }
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)
        print(f"Model saved to {filepath}")
    
    def load_weights(self, filepath: str):
        """Load model weights from file"""
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
        
        self.conv1.W = weights['conv1_W']
        self.conv1.b = weights['conv1_b']
        self.fc1.W = weights['fc1_W']
        self.fc1.b = weights['fc1_b']
        self.fc2.W = weights['fc2_W']
        self.fc2.b = weights['fc2_b']
        print(f"Model loaded from {filepath}")