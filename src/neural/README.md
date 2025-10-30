# Understanding Convolutional Neural Networks (CNN) - A Beginner's Guide

This README explains how our handmade CNN works for recognizing handwritten digits (0-9) from the MNIST dataset. Everything is built from scratch using only NumPy!

---

## Table of Contents
1. [What is a Neural Network?](#what-is-a-neural-network)
2. [What Makes a CNN Special?](#what-makes-a-cnn-special)
3. [Our CNN Architecture](#our-cnn-architecture)
4. [Layer-by-Layer Explanation](#layer-by-layer-explanation)
5. [How Training Works](#how-training-works)
6. [Code Walkthrough](#code-walkthrough)
7. [Running the Code](#running-the-code)

---

## What is a Neural Network?

Think of a neural network as a **pattern recognition machine**. Just like your brain learns to recognize faces, letters, or objects after seeing many examples, a neural network learns patterns from data.

### The Basic Idea
- **Input**: An image of a handwritten digit (e.g., a picture of "7")
- **Processing**: The network examines different features (curves, lines, edges)
- **Output**: A prediction ("I think this is a 7")

### How Does It "Learn"?
The network starts with random guesses. Then:
1. It makes a prediction
2. Checks how wrong it was (the "loss")
3. Adjusts its internal parameters slightly to do better next time
4. Repeats thousands of times until it gets good at recognizing digits

This process is called **training**.

---

## What Makes a CNN Special?

Regular neural networks treat images as flat lists of pixels, losing spatial information. **Convolutional Neural Networks (CNNs)** are designed specifically for images because they:

### 1. **Preserve Spatial Structure**
Images are 2D grids of pixels. CNNs keep this structure instead of flattening everything.

### 2. **Learn Local Patterns**
Instead of looking at the whole image at once, CNNs use small "windows" (filters) that scan across the image looking for local patterns like:
- Horizontal edges
- Vertical edges  
- Curves
- Corners

### 3. **Build Hierarchies of Features**
- **Early layers**: detect simple patterns (edges, lines)
- **Middle layers**: combine edges into shapes (curves, angles)
- **Later layers**: combine shapes into concepts (loops, straight lines → digits)

### 4. **Are Efficient**
The same filter is reused across the whole image, so CNNs need fewer parameters than fully-connected networks.

---

## Our CNN Architecture

Our network processes 28×28 pixel grayscale images through several layers:

```
Input Image (28×28)
      ↓
[ Convolution Layer ] ← Detects edges and patterns
      ↓
[ ReLU Activation ] ← Adds non-linearity
      ↓
[ Max Pooling ] ← Reduces size, keeps important info
      ↓
[ Flatten ] ← Converts 2D to 1D
      ↓
[ Dense Layer (128 neurons) ] ← Learns combinations
      ↓
[ ReLU Activation ]
      ↓
[ Dense Layer (10 neurons) ] ← One per digit (0-9)
      ↓
[ Softmax ] ← Converts to probabilities
      ↓
Output: Digit prediction
```

**Flow Summary:**
- **Input**: 28×28 grayscale image (784 pixels)
- **After Conv**: 8 feature maps of 26×26 (detected patterns)
- **After Pool**: 8 feature maps of 13×13 (downsampled)
- **After Flatten**: 1,352 numbers in a row
- **After Dense1**: 128 numbers (learned features)
- **Output**: 10 probabilities (one for each digit 0-9)

---

## Layer-by-Layer Explanation

### 1. **Convolution Layer** (`Conv2D`)

**What it does:** Scans the image with small "filters" to detect patterns.

**How it works:**
- Uses 8 filters, each 3×3 pixels
- Each filter slides across the image (like a magnifying glass)
- At each position, multiplies filter values with image values
- Produces a "feature map" showing where patterns were found

**Example:**
```
Original image patch:    Filter:           Result:
[1, 2, 1]                [1, 0, -1]        
[0, 1, 0]    *           [1, 0, -1]    =   Detects
[1, 2, 1]                [1, 0, -1]        vertical edges!
```

**Why it matters:** Different filters learn to detect different features (horizontal lines, curves, corners, etc.)

**Math:**
- Input: (batch_size, 1, 28, 28)
- Weights: (8, 1, 3, 3) - 8 filters, each 3×3
- Output: (batch_size, 8, 26, 26) - 8 feature maps

---

### 2. **ReLU Activation**

**What it does:** Adds "non-linearity" to help the network learn complex patterns.

**How it works:**
```python
ReLU(x) = max(0, x)
```
- If a number is positive, keep it
- If negative, make it zero

**Example:**
```
Input:  [-2, 0.5, -1, 3]
Output: [0, 0.5, 0, 3]
```

**Why it matters:** Without ReLU, the network could only learn straight-line patterns. ReLU allows it to learn curves, angles, and complex shapes.

---

### 3. **Max Pooling Layer** (`MaxPool2D`)

**What it does:** Reduces the size of feature maps while keeping the most important information.

**How it works:**
- Takes 2×2 windows across the feature map
- Keeps only the maximum value from each window
- Reduces dimensions by half

**Example:**
```
Input (4×4):          Output (2×2):
[1, 3, 2, 4]          [3, 4]
[2, 3, 1, 0]    →     [5, 8]
[0, 5, 1, 8]
[1, 2, 3, 1]
```

**Why it matters:** 
- Makes the network faster (fewer numbers to process)
- Makes pattern detection more robust (small shifts don't matter)
- Keeps only the strongest signals

---

### 4. **Flatten Layer**

**What it does:** Converts 2D feature maps into a 1D list.

**How it works:**
```
Input: 8 feature maps of 13×13
Output: One list of 1,352 numbers (8 × 13 × 13)
```

**Why it matters:** Fully-connected layers need 1D input, so we reshape the 2D data.

---

### 5. **Dense (Fully-Connected) Layer**

**What it does:** Learns combinations of features to make final decisions.

**How it works:**
- Every input connects to every output
- Each connection has a "weight" (importance)
- Computes: `output = (input × weights) + bias`

**First Dense Layer:**
- Input: 1,352 features
- Output: 128 learned representations
- Combines low-level features into high-level concepts

**Second Dense Layer (Output):**
- Input: 128 features
- Output: 10 scores (one per digit 0-9)
- Higher score = more confident about that digit

---

### 6. **Softmax + Cross-Entropy Loss**

**What it does:** Converts scores into probabilities and measures how wrong the prediction is.

**Softmax:**
Converts raw scores into probabilities that sum to 1.
```
Scores:        [2.1, 0.5, -1.2, 3.5, ...]
Probabilities: [0.14, 0.03, 0.01, 0.62, ...]
               ↑ These sum to 1.0
```

**Cross-Entropy Loss:**
Measures the difference between prediction and true label.
- If we predict correctly with high confidence: **low loss** (good!)
- If we predict incorrectly: **high loss** (bad!)

**Example:**
```
True label: 7
Prediction: [0.01, 0.02, ..., 0.85, ..., 0.01]  (digit 7 has 85%)
Loss: Low! (good prediction)

Prediction: [0.01, 0.02, ..., 0.10, ..., 0.70]  (digit 3 has 70%)
Loss: High! (wrong prediction)
```

---

## How Training Works

Training is the process of adjusting the network's weights so it makes better predictions. This happens through **backpropagation** and **gradient descent**.

### The Training Loop

```
FOR each epoch:
    FOR each batch of images:
        1. Forward Pass: Run images through network → get predictions
        2. Compute Loss: How wrong were the predictions?
        3. Backward Pass: Calculate gradients (which direction to adjust weights)
        4. Update Weights: Adjust weights slightly to reduce loss
```

### 1. **Forward Pass**
- Send an image through all layers
- Get a prediction (e.g., "80% sure it's a 7")

### 2. **Loss Calculation**
- Compare prediction to true label
- Compute a number measuring "wrongness" (loss)
- Goal: minimize this loss

### 3. **Backward Pass (Backpropagation)**
This is where the "learning" happens!

**The Chain Rule Intuition:**
Imagine you're hiking and want to go downhill (minimize loss). You need to know:
- Which direction is downhill? (gradient)
- How steep is it? (gradient magnitude)

Backpropagation uses calculus (the chain rule) to compute how each weight contributed to the error, working backwards through the network:

```
Output Layer → Dense Layer → Flatten → Pool → ReLU → Conv → Input
     ↑            ↑            ↑        ↑       ↑      ↑
 "This weight   "These      "These   [gradient flows backward]
  caused 0.3    weights      weights
  of error"     caused..."   caused..."
```

### 4. **Weight Update (Gradient Descent)**

Once we know which direction makes loss worse, we go the opposite way!

```python
new_weight = old_weight - (learning_rate × gradient)
```

- **Gradient**: Direction that increases loss
- **Learning rate**: How big of a step to take (e.g., 0.01)
- **Update**: Small step in the direction that decreases loss

**Analogy:** 
Imagine you're blindfolded on a hill and want to reach the bottom:
1. Feel which direction is downhill (gradient)
2. Take a small step that way (learning_rate × gradient)
3. Repeat until you reach the bottom (minimum loss)

### Training Progress Example

```
Epoch 1, Batch 0:    Loss = 2.30  (random guessing)
Epoch 1, Batch 100:  Loss = 0.85  (learning patterns)
Epoch 1, Batch 500:  Loss = 0.40  (getting better!)
Epoch 1 Complete:    Test Accuracy = 94.03%

Epoch 2, Batch 0:    Loss = 0.45
Epoch 2, Batch 500:  Loss = 0.28
Epoch 2 Complete:    Test Accuracy = 95.85%
```

The loss decreases and accuracy increases as the network learns!

---

## Code Walkthrough

### File Structure
```
neurosymbolic_mvp/src/neural/
├── cnn.py           ← Layer definitions (Conv2D, MaxPool2D, etc.)
├── model.py         ← CNN architecture (wires layers together)
├── mnist_loader.py  ← Data loading utilities
└── train.py         ← Training script
```

---

### `cnn.py` - Building Blocks

This file contains all the layer classes. Each layer has:
- `__init__()`: Initialize parameters (weights, biases)
- `forward()`: Process input → output
- `backward()`: Compute gradients during training

**Key Classes:**

**`Conv2D`**
```python
Conv2D(in_channels=1, out_channels=8, kernel_size=3)
# Creates 8 filters, each 3×3, for detecting patterns
```

**`MaxPool2D`**
```python
MaxPool2D()  # Default: 2×2 window, stride 2
# Reduces spatial dimensions by half
```

**`Dense`**
```python
Dense(in_features=1352, out_features=128)
# Fully-connected: every input connects to every output
```

**`ReLU`**
```python
ReLU()  # Activation: max(0, x)
```

**`SoftmaxCrossEntropy`**
```python
SoftmaxCrossEntropy()
# Combined softmax + loss function
```

---

### `model.py` - The Complete CNN

This file defines the `CNN` class that chains all layers together.

**Architecture:**
```python
class CNN:
    def __init__(self):
        self.conv1 = Conv2D(1, 8, 3)      # Convolution
        self.relu1 = ReLU()               # Activation
        self.pool1 = MaxPool2D()          # Pooling
        self.flatten = Flatten()          # Reshape
        self.fc1 = Dense(1352, 128)       # Dense layer 1
        self.relu2 = ReLU()               # Activation
        self.fc2 = Dense(128, 10)         # Output layer
        self.loss_fn = SoftmaxCrossEntropy()
```

**Key Methods:**

**`forward(x)`** - Make predictions
```python
def forward(self, x):
    x = self.conv1.forward(x)   # (N, 1, 28, 28) → (N, 8, 26, 26)
    x = self.relu1.forward(x)   # Apply activation
    x = self.pool1.forward(x)   # (N, 8, 26, 26) → (N, 8, 13, 13)
    x = self.flatten.forward(x) # (N, 8, 13, 13) → (N, 1352)
    x = self.fc1.forward(x)     # (N, 1352) → (N, 128)
    x = self.relu2.forward(x)   # Apply activation
    x = self.fc2.forward(x)     # (N, 128) → (N, 10)
    return x  # Logits (raw scores)
```

**`backward()`** - Compute gradients
```python
def backward(self):
    grad = self.loss_fn.backward()  # Start from loss
    grad = self.fc2.backward(grad)  # Backprop through each layer
    grad = self.relu2.backward(grad)
    # ... (continues backward through all layers)
```

**`update_weights(learning_rate)`** - Learn from gradients
```python
def update_weights(self, learning_rate=0.01):
    for layer in self.trainable_layers:
        layer.W -= learning_rate * layer.grad_W  # SGD update
        layer.b -= learning_rate * layer.grad_b
```

---

### `mnist_loader.py` - Data Loading

**`load_mnist_npz(path)`**
- Loads MNIST from `.npz` file
- Normalizes pixels to [0, 1] range
- Reshapes to (N, 1, 28, 28) for CNN
- Returns: train/test images and labels

**`get_batch(x, y, batch_size)`**
- Samples random batch for training
- Returns: (batch_images, batch_labels)

---

### `train.py` - Training Script

**Main function: `train_cnn()`**

```python
def train_cnn(epochs=5, batch_size=32, learning_rate=0.01):
    # 1. Load data
    x_train, y_train, x_test, y_test = load_mnist_npz(...)
    
    # 2. Create model
    model = CNN()
    
    # 3. Training loop
    for epoch in range(epochs):
        for batch_idx in range(batches_per_epoch):
            # Get batch
            x_batch, y_batch = get_batch(x_train, y_train, batch_size)
            
            # Forward pass
            logits = model.forward(x_batch)
            
            # Compute loss
            loss = model.compute_loss(logits, y_batch)
            
            # Backward pass
            model.backward()
            
            # Update weights
            model.update_weights(learning_rate)
        
        # Evaluate after each epoch
        accuracy = evaluate(model, x_test, y_test)
        print(f"Test Accuracy: {accuracy:.2f}%")
```

**Key Parameters:**
- **`epochs`**: How many times to go through entire dataset (default: 5)
- **`batch_size`**: How many images per update (default: 32)
- **`learning_rate`**: Step size for weight updates (default: 0.01)

---

## Running the Code

### Prerequisites
```bash
# Only numpy is required!
pip install numpy
```

### Download MNIST Data
```python
import numpy as np
from urllib.request import urlretrieve

urlretrieve(
    'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz',
    'mnist_data.npz'
)
```

### Train the Model
```bash
cd neurosymbolic_mvp/src/neural
python train.py
```

### Expected Output
```
Loading MNIST data...
Training samples: 60000, Test samples: 10000
Initializing CNN model...

Starting training for 5 epochs...
  Epoch 1/5, Batch 0/1875, Loss: 2.3991
  Epoch 1/5, Batch 100/1875, Loss: 0.8542
  ...
Epoch 1/5 completed. Average Loss: 0.3784
  Evaluating on test set...
  Test Accuracy: 94.03%

  Epoch 2/5, Batch 0/1875, Loss: 0.4477
  ...
Epoch 2/5 completed. Average Loss: 0.1857
  Test Accuracy: 95.85%
```

---

## Key Concepts Summary

### 1. **Convolution**
- Sliding window that detects patterns
- Learns filters automatically during training
- Preserves spatial structure of images

### 2. **Activation (ReLU)**
- Adds non-linearity: `max(0, x)`
- Allows network to learn complex patterns
- Simple but effective

### 3. **Pooling**
- Reduces size while keeping important info
- Makes network faster and more robust
- Takes maximum value in each window

### 4. **Dense/Fully-Connected**
- Learns combinations of features
- Every input connects to every output
- Makes final classification decision

### 5. **Backpropagation**
- Uses chain rule to compute gradients
- Flows backward through network
- Tells us how to adjust each weight

### 6. **Gradient Descent**
- Iteratively adjusts weights to minimize loss
- Takes small steps in direction that reduces error
- Learning rate controls step size

---

## What Makes This Implementation Special?

### ✅ **Educational Value**
- No "magic" black-box libraries
- Every operation is explicit and visible
- You can step through with a debugger

### ✅ **From Scratch**
- Only uses NumPy (basic array operations)
- No PyTorch, TensorFlow, or Keras
- Implements forward AND backward passes manually

### ✅ **Clear Architecture**
- Modular design (one class per layer)
- Easy to modify and experiment
- Well-commented code

### ✅ **Real Performance**
- Achieves ~96% accuracy on MNIST
- Comparable to library implementations
- Proves you understand how it works!

---

## Going Further

Now that you understand CNNs, you can:

1. **Experiment with architecture:**
   - Add more convolutional layers
   - Try different filter sizes
   - Change number of filters

2. **Try different datasets:**
   - Fashion-MNIST (clothing images)
   - CIFAR-10 (colored images)
   - Your own image data

3. **Optimize the code:**
   - Vectorize operations further
   - Implement batch normalization
   - Try different optimizers (Adam, RMSprop)

4. **Add advanced features:**
   - Dropout for regularization
   - Data augmentation
   - Learning rate scheduling

---

## Common Questions

**Q: Why do we need multiple layers?**
A: Each layer learns increasingly complex patterns:
- Layer 1: edges, corners
- Layer 2: shapes, curves  
- Layer 3: parts of digits
- Output: complete digits

**Q: What are "weights" exactly?**
A: Numbers that determine how much influence one neuron has on another. Training = finding good weight values.

**Q: Why normalize pixel values to [0, 1]?**
A: Makes training more stable and faster. Large input values can cause numerical issues.

**Q: What is a "batch"?**
A: A small group of examples processed together. More efficient than processing one at a time.

**Q: How long does training take?**
A: On CPU: ~10-20 minutes for 5 epochs. On GPU: ~1-2 minutes.

**Q: Can I use this for other tasks?**
A: Yes! The principles are the same. Just adjust the input/output layers for your data.

---

## Congratulations! 🎉

You now understand how a CNN works from the ground up. This is the same technology used in:
- Self-driving cars (detecting objects)
- Medical imaging (finding diseases)
- Face recognition (unlocking phones)
- Image search (Google Images)

You didn't just use a library—you built one yourself!

---

## Next Steps: Neuro-Symbolic Integration

This CNN is the **neural** part of our neurosymbolic system. Next, we'll add:

1. **Symbolic reasoning layer**: Logic rules for digit arithmetic
2. **Bridge layer**: Convert CNN outputs to symbolic facts
3. **Feedback loop**: Use logic to correct neural mistakes

Stay tuned! 🚀
