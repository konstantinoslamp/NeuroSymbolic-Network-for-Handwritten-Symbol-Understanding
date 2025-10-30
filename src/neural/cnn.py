import numpy as np
from typing import Tuple, List

class Conv2D:
    """
    A 2D convolution layer (stride=1, no padding).
    Args:
      in_channels: number of input feature maps
      out_channels: number of filters
      kernel_size: size of the (square) convolution kernel
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        # initialize weights W: (out_channels, in_channels, K, K) and biases b: (out_channels,)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        #Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (N, in_channels, H, W)
        self.x = x  # cache for backward
        N, C, H, W = x.shape
        K = self.kernel_size
        OC = self.out_channels
        OH = H - K + 1
        OW = W - K + 1
        out = np.zeros((N, OC, OH, OW))
        
        # Optimized: batch process spatial positions
        for n in range(N):
            for i in range(OH):
                for j in range(OW):
                    # Extract window: (C, K, K)
                    window = x[n, :, i:i+K, j:j+K]
                    # Vectorized: (OC, C, K, K) * (C, K, K) -> sum over C,K,K -> (OC,)
                    out[n, :, i, j] = np.sum(self.W * window, axis=(1, 2, 3)) + self.b
        return out

    # In Conv2D class, fix the backward method indentation:
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        Compute gradients for convolution layer.
        grad_out: gradient from next layer, shape (N, out_channels, OH, OW)
        returns: gradient w.r.t. input, shape (N, in_channels, H, W)
        """
        N, OC, OH, OW = grad_out.shape
        _, C, H, W = self.x.shape
        K = self.kernel_size
        
        # Initialize gradients
        grad_x = np.zeros_like(self.x)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros(OC)
        
        # Compute gradients (simplified version)
        for n in range(N):
            for oc in range(OC):
                for i in range(OH):
                    for j in range(OW):
                        for ic in range(C):
                            window = self.x[n, ic, i:i+K, j:j+K]
                            self.grad_W[oc, ic] += window * grad_out[n, oc, i, j]
                            grad_x[n, ic, i:i+K, j:j+K] += self.W[oc, ic] * grad_out[n, oc, i, j]
                self.grad_b[oc] += np.sum(grad_out[:, oc, :, :])
        
        return grad_x

class MaxPool2D:
    """
    A 2x2 max-pooling layer (stride=2).
    """
    def __init__(self):
        self.kernel_size = 2
        self.stride = 2

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.input_shape = x.shape  # cache for backward
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride
        OH, OW = H // SH, W // SW
        out = np.zeros((N, C, OH, OW))
        for n in range(N):
            for c in range(C):
                for i in range(OH):
                    for j in range(OW):
                        h_start, w_start = i * SH, j * SW
                        window = x[n, c, h_start:h_start+KH, w_start:w_start+KW]
                        out[n, c, i, j] = np.max(window)
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        Gradient flows only through the max element positions.
        """
        N, C, OH, OW = grad_out.shape
        grad_x = np.zeros(self.input_shape)
        
        for n in range(N):
            for c in range(C):
                for i in range(OH):
                    for j in range(OW):
                        h_start, w_start = i * self.stride, j * self.stride
                        window = self.x[n, c, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]
                        max_val = np.max(window)
                        mask = (window == max_val)
                        grad_x[n, c, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size] += mask * grad_out[n, c, i, j]
        
        return grad_x

class Flatten:
    """Flattens a rank-4 tensor (N, C, H, W) → (N, C·H·W)"""
    def forward(self, x: np.ndarray) -> np.ndarray:
        #Save shape for backward
        self.input_shape = x.shape
        N = x.shape[0]
        return x.reshape(N, -1)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        #Restore original shape
        return grad_out.reshape(self.input_shape)

class Dense:
    """Fully-connected layer"""
    def __init__(self, in_features: int, out_features: int):
        # Xavier initialization
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)
        self.x = None  # cache input for backward

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (N, in_features)
        self.x = x  # cache for backward
        return x @ self.W + self.b  # (N, out_features)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # grad_out: (N, out_features)
        # Compute gradients for weights, biases, and input
        grad_x = grad_out @ self.W.T  # (N, in_features)
        grad_W = self.x.T @ grad_out  # (in_features, out_features)
        grad_b = np.sum(grad_out, axis=0)  # (out_features,)
        # Store gradients for optimizer (if needed)
        self.grad_W = grad_W
        self.grad_b = grad_b
        return grad_x

class ReLU:
    """Element-wise ReLU activation"""
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0  # cache for backward
        return np.maximum(0, x)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # Gradient flows only where input > 0
        return grad_out * self.mask

class SoftmaxCrossEntropy:
    """
    Combined softmax + cross-entropy loss.
    forward(logits, labels) → scalar loss
    backward() → gradient on logits
    """
    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        # logits: (N, num_classes), labels: (N,) with integer class indices
        self.logits = logits
        self.labels = labels
        # Stable softmax
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        N = logits.shape[0]
        # Cross-entropy loss
        correct_logprobs = -np.log(self.probs[np.arange(N), labels] + 1e-12)
        loss = np.sum(correct_logprobs) / N
        return loss

    def backward(self) -> np.ndarray:
        # Gradient of loss w.r.t. logits
        N = self.logits.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.labels] -= 1
        grad /= N
        return grad